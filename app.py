"""
Streamlit app — Early Student Dropout Risk Assessment
Run: streamlit run app.py
Requires model_artifacts/ produced by SUMMER_PROJECT_ROBUST.ipynb.
"""
import os
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Student Dropout Risk',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='expanded',
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_artifacts')

TIER_STYLE = {
    'HIGH':   {'bg': '#fde8e8', 'border': '#e74c3c', 'text': '#c0392b', 'icon': '🔴'},
    'MEDIUM': {'bg': '#fef9e7', 'border': '#f39c12', 'text': '#d35400', 'icon': '🟡'},
    'LOW':    {'bg': '#e9f7ef', 'border': '#27ae60', 'text': '#1e8449', 'icon': '🟢'},
}

INTERVENTIONS = {
    'Grade_1':               {'check': lambda v: v <= 8,  'msg': 'Very low Grade 1 — assign a peer tutor before semester 2 begins.'},
    'Grade_2':               {'check': lambda v: v <= 8,  'msg': 'Very low Grade 2 — schedule urgent academic support; consider remedial classes.'},
    'Grade_Trend':           {'check': lambda v: v < -2,  'msg': 'Grades declining significantly — investigate cause (personal / health / social) with a counselor.'},
    'Low_Grade_Flag':        {'check': lambda v: v == 1,  'msg': 'Scored ≤5 in at least one semester — flag for intensive academic intervention.'},
    'Number_of_Absences':    {'check': lambda v: v > 10,  'msg': 'High absenteeism — contact parents and set up an attendance monitoring plan.'},
    'Number_of_Failures':    {'check': lambda v: v >= 2,  'msg': '2+ subject failures — refer to remedial classes and set monthly progress reviews.'},
    'Study_Time':            {'check': lambda v: v <= 1,  'msg': 'Very low study time — enroll in a structured study-skills workshop.'},
    'Total_Alcohol':         {'check': lambda v: v >= 6,  'msg': 'High combined alcohol consumption — refer to school counselor for a welfare check.'},
    'Wants_Higher_Education': {'check': lambda v: str(v).strip().lower() == 'no', 'msg': 'No aspiration for higher education — provide career counseling and motivational support.'},
    'Academic_Risk':         {'check': lambda v: v >= 3,  'msg': 'High Academic Risk score (failures + absences) — prioritise for weekly check-ins.'},
    'Health_Status':         {'check': lambda v: v <= 2,  'msg': 'Poor self-reported health — connect with school health services.'},
    'Family_Relationship':   {'check': lambda v: v <= 2,  'msg': 'Poor family relationships — involve school social worker or family support programme.'},
    'Going_Out':             {'check': lambda v: v >= 4,  'msg': 'Spending a lot of time going out — discuss time management and study–life balance.'},
}

# Base input columns (before feature engineering).
BASE_CAT_COLS = [
    'School', 'Gender', 'Address', 'Family_Size', 'Parental_Status',
    'Mother_Job', 'Father_Job', 'Reason_for_Choosing_School', 'Guardian',
    'School_Support', 'Family_Support', 'Extra_Paid_Class',
    'Extra_Curricular_Activities', 'Attended_Nursery', 'Wants_Higher_Education',
    'Internet_Access', 'In_Relationship',
]
BASE_NUM_COLS = [
    'Age', 'Mother_Education', 'Father_Education', 'Travel_Time', 'Study_Time',
    'Number_of_Failures', 'Family_Relationship', 'Free_Time', 'Going_Out',
    'Weekend_Alcohol_Consumption', 'Weekday_Alcohol_Consumption',
    'Health_Status', 'Number_of_Absences', 'Grade_1', 'Grade_2',
]


# ── Artifact loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading model…')
def load_artifacts():
    if not os.path.isdir(ARTIFACTS_DIR):
        return None
    cal      = joblib.load(os.path.join(ARTIFACTS_DIR, 'calibrated_pipeline.joblib'))
    fin      = joblib.load(os.path.join(ARTIFACTS_DIR, 'final_pipeline.joblib'))
    meta     = joblib.load(os.path.join(ARTIFACTS_DIR, 'metadata.joblib'))
    clf      = fin.named_steps['clf']
    explainer = shap.TreeExplainer(clf)
    # X_columns may not exist in older saves; fall back to cat + num order
    if 'X_columns' not in meta:
        meta['X_columns'] = meta['categorical_cols'] + meta['numerical_cols']
    return cal, fin, meta, explainer


# ── Feature engineering (must mirror the notebook exactly) ───────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Grade_Avg']        = (df['Grade_1'] + df['Grade_2']) / 2
    df['Grade_Trend']      = df['Grade_2'] - df['Grade_1']
    df['Low_Grade_Flag']   = ((df['Grade_1'] <= 5) | (df['Grade_2'] <= 5)).astype(int)
    df['Total_Alcohol']    = df['Weekend_Alcohol_Consumption'] + df['Weekday_Alcohol_Consumption']
    df['Max_Parent_Edu']   = df[['Mother_Education', 'Father_Education']].max(axis=1)
    df['Academic_Risk']    = df['Number_of_Failures'] * 3 + df['Number_of_Absences'] / 5.0
    df['Study_Efficiency'] = df['Grade_Avg'] / (df['Study_Time'] + 1e-3)
    return df


def prepare_df(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = engineer_features(df)
    df = df.drop(columns=[c for c in ['Grade_Avg', 'Final_Grade', 'Dropped_Out'] if c in df.columns])
    cat_cols = meta['categorical_cols']
    X_cols   = meta['X_columns']
    for col in X_cols:
        if col not in df.columns:
            df[col] = 'unknown' if col in cat_cols else 0
    df = df[[c for c in X_cols if c in df.columns]]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# ── Interventions ─────────────────────────────────────────────────────────────
def get_interventions(raw: dict, sv: np.ndarray, feat_names: list, top_n: int = 8) -> list:
    top_idx = np.argsort(sv)[::-1][:top_n]
    recs, seen = [], set()
    for i in top_idx:
        if sv[i] <= 0:
            continue
        fn = feat_names[i]
        for feat, rule in INTERVENTIONS.items():
            if feat in seen:
                continue
            if fn.startswith(feat):
                val = raw.get(feat)
                if val is None:
                    break
                try:
                    if rule['check'](val):
                        recs.append({'Feature': feat, 'Value': val,
                                     'SHAP': round(float(sv[i]), 3), 'Action': rule['msg']})
                        seen.add(feat)
                except Exception:
                    pass
                break
    return recs


# ── Prediction helpers ────────────────────────────────────────────────────────
def _tier(prob: float, meta: dict) -> str:
    if prob >= meta['high_risk_threshold']:
        return 'HIGH'
    if prob >= meta['best_thr_cal']:
        return 'MEDIUM'
    return 'LOW'


def predict_with_shap(student_dict: dict, cal, fin, meta, explainer) -> dict:
    df   = prepare_df(pd.DataFrame([student_dict]), meta)
    prob = float(cal.predict_proba(df)[0, 1])
    tier = _tier(prob, meta)

    proc    = fin.named_steps['preprocessor'].transform(df)
    proc_df = pd.DataFrame(np.asarray(proc), columns=meta['feature_names'])
    sv_raw  = np.asarray(explainer(proc_df).values)[0]
    sv      = sv_raw[:, 1] if sv_raw.ndim == 2 else sv_raw

    recs = get_interventions(student_dict, sv, meta['feature_names'])
    return {
        'probability': prob,
        'tier': tier,
        'at_risk': prob >= meta['best_thr_cal'],
        'shap': sv,
        'recommendations': recs,
    }


def predict_batch(rows: list[dict], cal, meta) -> list[dict]:
    results = []
    for row in rows:
        df   = prepare_df(pd.DataFrame([row]), meta)
        prob = float(cal.predict_proba(df)[0, 1])
        results.append({
            'Dropout_Prob_%': round(prob * 100, 1),
            'Risk_Tier': _tier(prob, meta),
            'At_Risk': prob >= meta['best_thr_cal'],
        })
    return results


# ── Charts ────────────────────────────────────────────────────────────────────
def shap_bar_fig(sv: np.ndarray, feature_names: list, top_n: int = 12) -> plt.Figure:
    idx   = np.argsort(np.abs(sv))[::-1][:top_n]
    feats = [feature_names[i].replace('ohe__', '').replace('scaler__', '') for i in idx]
    vals  = sv[idx]
    clrs  = ['#e74c3c' if v > 0 else '#3498db' for v in vals]

    fig, ax = plt.subplots(figsize=(6, max(3.5, top_n * 0.38)))
    ax.barh(range(len(feats)), vals[::-1], color=clrs[::-1], edgecolor='grey', linewidth=0.3)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats[::-1], fontsize=8)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_xlabel('SHAP value', fontsize=9)
    ax.set_title('Risk drivers  (red = increases risk, blue = reduces risk)', fontsize=9)
    plt.tight_layout()
    return fig


def tier_bar_fig(counts: dict) -> plt.Figure:
    order  = ['HIGH', 'MEDIUM', 'LOW']
    labels = [k for k in order if k in counts]
    values = [counts[k] for k in labels]
    colors = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'}

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, color=[colors[k] for k in labels], edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val),
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Students')
    ax.set_title('Risk Tier Distribution')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_single(cal, fin, meta, explainer):
    st.header('Single Student Risk Assessment')
    st.caption(
        'Fill in the student profile and click **Assess Risk**. '
        'Results include a calibrated dropout probability, the main contributing factors, '
        'and suggested interventions.'
    )

    with st.form('student_form'):
        tab_ac, tab_pe, tab_fa, tab_bh = st.tabs(['📚 Academic', '👤 Personal', '👨‍👩‍👧 Family', '🎭 Behaviour'])

        with tab_ac:
            c1, c2, c3 = st.columns(3)
            school         = c1.selectbox('School', ['GP', 'MS'])
            grade1         = c1.slider('Grade 1  (0–20)', 0, 20, 10)
            grade2         = c1.slider('Grade 2  (0–20)', 0, 20, 10)
            failures       = c2.selectbox('Past subject failures', [0, 1, 2, 3])
            studytime      = c2.selectbox('Weekly study time', [1, 2, 3, 4],
                                          format_func=lambda x: {1:'< 2 h', 2:'2–5 h', 3:'5–10 h', 4:'> 10 h'}[x])
            absences       = c2.number_input('School absences', 0, 93, 4)
            reason         = c3.selectbox('Reason for choosing school', ['course', 'home', 'reputation', 'other'])
            school_support = c3.selectbox('Extra support from school', ['no', 'yes'])
            extra_paid     = c3.selectbox('Extra paid subject classes', ['no', 'yes'])
            extra_curr     = c3.selectbox('Extra-curricular activities', ['no', 'yes'])

        with tab_pe:
            c1, c2 = st.columns(2)
            gender       = c1.selectbox('Gender', ['F', 'M'])
            age          = c1.slider('Age', 15, 22, 16)
            address      = c1.selectbox('Home address', ['U', 'R'],
                                         format_func=lambda x: {'U': 'Urban', 'R': 'Rural'}[x])
            internet     = c2.selectbox('Internet access at home', ['yes', 'no'])
            relationship = c2.selectbox('Currently in a relationship', ['no', 'yes'])
            higher_edu   = c2.selectbox('Wants higher education', ['yes', 'no'])

        with tab_fa:
            c1, c2, c3 = st.columns(3)
            fam_size    = c1.selectbox('Family size', ['GT3', 'LE3'],
                                        format_func=lambda x: {'GT3': '> 3 members', 'LE3': '≤ 3 members'}[x])
            par_status  = c1.selectbox('Parents living together', ['T', 'A'],
                                        format_func=lambda x: {'T': 'Together', 'A': 'Apart'}[x])
            guardian    = c1.selectbox('Primary guardian', ['mother', 'father', 'other'])
            nursery     = c1.selectbox('Attended nursery school', ['yes', 'no'])
            mom_edu     = c2.selectbox('Mother education  (0=none … 4=higher)', [0, 1, 2, 3, 4])
            dad_edu     = c2.selectbox('Father education  (0=none … 4=higher)', [0, 1, 2, 3, 4])
            mom_job     = c2.selectbox('Mother occupation', ['at_home', 'health', 'other', 'services', 'teacher'])
            dad_job     = c3.selectbox('Father occupation', ['at_home', 'health', 'other', 'services', 'teacher'])
            fam_rel     = c3.selectbox('Family relationship  (1=very bad … 5=excellent)', [1, 2, 3, 4, 5])
            fam_support = c3.selectbox('Family educational support', ['yes', 'no'])
            travel      = c3.selectbox('Travel time to school', [1, 2, 3, 4],
                                        format_func=lambda x: {1:'< 15 min', 2:'15–30 min', 3:'30–60 min', 4:'> 1 h'}[x])

        with tab_bh:
            c1, c2 = st.columns(2)
            free_time = c1.selectbox('Free time after school  (1=low … 5=high)', [1, 2, 3, 4, 5])
            going_out = c1.selectbox('Going out with friends  (1=low … 5=high)', [1, 2, 3, 4, 5])
            health    = c1.selectbox('Health status  (1=very bad … 5=excellent)', [1, 2, 3, 4, 5])
            wkdy_alc  = c2.selectbox('Weekday alcohol  (1=low … 5=high)', [1, 2, 3, 4, 5])
            wknd_alc  = c2.selectbox('Weekend alcohol  (1=low … 5=high)', [1, 2, 3, 4, 5])

        submitted = st.form_submit_button('🔍  Assess Risk', use_container_width=True, type='primary')

    if not submitted:
        st.info('Complete the form above and click **Assess Risk**.')
        return

    student = {
        'School': school, 'Gender': gender, 'Age': age, 'Address': address,
        'Family_Size': fam_size, 'Parental_Status': par_status,
        'Mother_Education': mom_edu, 'Father_Education': dad_edu,
        'Mother_Job': mom_job, 'Father_Job': dad_job,
        'Reason_for_Choosing_School': reason, 'Guardian': guardian,
        'Travel_Time': travel, 'Study_Time': studytime,
        'Number_of_Failures': failures,
        'School_Support': school_support, 'Family_Support': fam_support,
        'Extra_Paid_Class': extra_paid, 'Extra_Curricular_Activities': extra_curr,
        'Attended_Nursery': nursery, 'Wants_Higher_Education': higher_edu,
        'Internet_Access': internet, 'In_Relationship': relationship,
        'Family_Relationship': fam_rel, 'Free_Time': free_time, 'Going_Out': going_out,
        'Weekend_Alcohol_Consumption': wknd_alc, 'Weekday_Alcohol_Consumption': wkdy_alc,
        'Health_Status': health, 'Number_of_Absences': int(absences),
        'Grade_1': grade1, 'Grade_2': grade2,
    }

    with st.spinner('Running prediction and SHAP analysis…'):
        result = predict_with_shap(student, cal, fin, meta, explainer)

    st.divider()
    tier = result['tier']
    sty  = TIER_STYLE[tier]
    prob_pct = result['probability'] * 100

    col_card, col_metrics, col_shap = st.columns([1.2, 1, 2])

    with col_card:
        st.markdown(
            f"""
            <div style="background:{sty['bg']};border:2px solid {sty['border']};
                        border-radius:12px;padding:24px;text-align:center;margin-top:6px">
                <div style="font-size:2.8rem;line-height:1">{sty['icon']}</div>
                <div style="font-size:1.7rem;font-weight:700;color:{sty['text']};
                            margin-top:8px">{tier} RISK</div>
                <div style="font-size:1.05rem;color:{sty['text']};margin-top:4px">
                    {prob_pct:.1f}% dropout probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_metrics:
        st.metric('Calibrated probability', f'{prob_pct:.1f}%')
        st.metric('Decision threshold', f"{meta['best_thr_cal'] * 100:.1f}%")
        st.metric('At-risk flag', '✅ Yes' if result['at_risk'] else '❌ No')

    with col_shap:
        fig = shap_bar_fig(result['shap'], meta['feature_names'])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.subheader('Recommended Interventions')
    recs = result['recommendations']
    if not recs:
        st.success('No high-priority interventions triggered for this student profile.')
    else:
        for rec in recs:
            with st.container(border=True):
                left, right = st.columns([1, 3])
                left.metric(rec['Feature'], rec['Value'], f"SHAP +{rec['SHAP']:.3f}")
                right.markdown(f"**→ {rec['Action']}**")

    st.caption(
        '⚠️ This prediction is decision support only. '
        'Always review high-risk flags with a qualified educator or counselor before taking action.'
    )


def page_batch(cal, meta):
    st.header('Batch Risk Assessment')
    st.caption(
        'Upload a CSV containing student records. '
        'Columns must match the base feature names. Missing columns are filled with neutral defaults.'
    )

    # Template download
    template_df = pd.DataFrame(columns=BASE_CAT_COLS + BASE_NUM_COLS)
    buf = io.BytesIO()
    template_df.to_csv(buf, index=False)
    st.download_button('⬇️  Download CSV template', buf.getvalue(), 'student_template.csv', 'text/csv')

    uploaded = st.file_uploader('Upload student CSV', type='csv')
    if uploaded is None:
        return

    raw_df = pd.read_csv(uploaded)
    st.subheader(f'Preview — {len(raw_df)} rows')
    st.dataframe(raw_df.head(10), use_container_width=True)

    if not st.button('🔍  Run Batch Prediction', type='primary'):
        return

    prog = st.progress(0, 'Running predictions…')
    rows = raw_df.to_dict(orient='records')
    results = []
    for i, row in enumerate(rows):
        df   = prepare_df(pd.DataFrame([row]), meta)
        prob = float(cal.predict_proba(df)[0, 1])
        results.append({
            'Dropout_Prob_%': round(prob * 100, 1),
            'Risk_Tier': _tier(prob, meta),
            'At_Risk': prob >= meta['best_thr_cal'],
        })
        prog.progress((i + 1) / len(rows))
    prog.empty()

    res_df = pd.DataFrame(results)
    out_df = pd.concat([res_df, raw_df.reset_index(drop=True)], axis=1)

    st.subheader('Results')

    def _row_color(row):
        c = {'HIGH': 'background-color:#fde8e8', 'MEDIUM': 'background-color:#fef9e7', 'LOW': ''}.get(row['Risk_Tier'], '')
        return [c] * len(row)

    st.dataframe(
        out_df.style.apply(_row_color, axis=1),
        use_container_width=True,
    )

    tier_counts = res_df['Risk_Tier'].value_counts().to_dict()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Total', len(out_df))
    c2.metric('🔴 HIGH',   tier_counts.get('HIGH', 0))
    c3.metric('🟡 MEDIUM', tier_counts.get('MEDIUM', 0))
    c4.metric('🟢 LOW',    tier_counts.get('LOW', 0))
    at_risk_pct = res_df['At_Risk'].mean() * 100
    c5.metric('At-risk rate', f'{at_risk_pct:.1f}%')

    fig = tier_bar_fig(tier_counts)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    csv_buf = io.BytesIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button('⬇️  Download results CSV', csv_buf.getvalue(), 'dropout_predictions.csv', 'text/csv')


def page_model_info(meta):
    st.header('Model Information')

    c1, c2, c3 = st.columns(3)
    c1.metric('Model',             meta.get('model_name', '—'))
    c2.metric('Imbalance strategy', meta.get('imbalance', '—'))
    c3.metric('CV F2-score',        f"{meta.get('cv_f2', 0):.4f}")

    st.subheader('Decision Thresholds')
    t1, t2, t3 = st.columns(3)
    t1.metric('Raw threshold (pre-calibration)', f"{meta.get('best_thr', 0):.2f}")
    t2.metric('Calibrated threshold',            f"{meta.get('best_thr_cal', 0):.2f}")
    t3.metric('High-risk cutoff',                f"{meta.get('high_risk_threshold', 0.65):.2f}")

    thr     = meta.get('best_thr_cal', 0)
    hr_thr  = meta.get('high_risk_threshold', 0.65)
    st.subheader('Risk Tier Definitions')
    st.markdown(
        f"""
| Tier | Calibrated probability | Suggested action |
|---|---|---|
| 🟢 LOW | below {thr:.0%} | No immediate action required |
| 🟡 MEDIUM | {thr:.0%} – {hr_thr:.0%} | Monitor; consider a pastoral check-in |
| 🔴 HIGH | {hr_thr:.0%} and above | Prioritise for counselling and academic support |
        """
    )

    with st.expander('Feature lists'):
        fc1, fc2 = st.columns(2)
        fc1.write('**Categorical features**')
        fc1.write(pd.Series(meta.get('categorical_cols', []), name='column'))
        fc2.write('**Numerical features (including engineered)**')
        fc2.write(pd.Series(meta.get('numerical_cols', []), name='column'))

    st.info(
        '**Important:** This model is intended for decision support only. '
        'High-risk predictions must be reviewed by a qualified educator or counselor '
        'before any intervention is made.'
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    artifacts = load_artifacts()

    with st.sidebar:
        st.title('🎓 Dropout Risk')
        st.caption('Early student dropout risk classification')
        st.divider()
        page = st.radio(
            'Navigation',
            ['Single Assessment', 'Batch Assessment', 'Model Info'],
            label_visibility='collapsed',
        )
        st.divider()
        if artifacts:
            meta = artifacts[2]
            st.caption(f"**Model:** {meta.get('model_name', '?')}")
            st.caption(f"**CV F2:** {meta.get('cv_f2', 0):.4f}")
            st.caption(f"**Calibrated thr:** {meta.get('best_thr_cal', 0):.2f}")
            st.caption(f"**High-risk thr:** {meta.get('high_risk_threshold', 0.65):.2f}")
        else:
            st.error('`model_artifacts/` not found.\nRun SUMMER_PROJECT_ROBUST.ipynb first.')

    if artifacts is None:
        st.error(
            'Model artifacts not found. '
            'Run **SUMMER_PROJECT_ROBUST.ipynb** to generate `model_artifacts/`, '
            'then restart the app.'
        )
        st.stop()

    cal, fin, meta, explainer = artifacts

    if page == 'Single Assessment':
        page_single(cal, fin, meta, explainer)
    elif page == 'Batch Assessment':
        page_batch(cal, meta)
    else:
        page_model_info(meta)


if __name__ == '__main__':
    main()

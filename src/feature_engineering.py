"""
feature_engineering.py - Derive meaningful features from raw data
================================================================
The raw dataset has weak correlations with attendance (~0.08 max).
This module creates richer features that capture actual patterns:

1. Temporal features  - month, weekend flag, semester week
2. Student history    - rolling attendance, streaks
3. Event popularity   - historical attendance rates per topic/speaker
4. Interaction terms  - combining features for better signals

These derived features should significantly improve model performance
because they encode the actual behavioral patterns that raw columns miss.
"""

import pandas as pd
import numpy as np


def add_temporal_features(df):
    """
    Extract time-based features from event_date.
    The idea is that attendance patterns change over the semester -
    students are more active early on and attendance drops near exams.
    """
    df = df.copy()
    
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['month'] = df['event_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    
    # semester start is roughly August for odd sem, January for even sem
    # calculate which week of the semester the event falls in
    def get_semester_week(date):
        month = date.month
        if month >= 8:  # odd semester (Aug-Dec)
            sem_start = pd.Timestamp(year=date.year, month=8, day=1)
        else:  # even semester (Jan-May)
            sem_start = pd.Timestamp(year=date.year, month=1, day=1)
        return max(1, (date - sem_start).days // 7 + 1)
    
    df['semester_week'] = df['event_date'].apply(get_semester_week)
    
    return df


def add_student_history_features(df):
    """
    Build student-level history features.
    
    Includes rolling attendance, streaks, consistency (std dev),
    and attendance recency for richer behavioral signals.
    """
    df = df.copy()
    df = df.sort_values(['student_id', 'event_date'])
    
    # rolling attendance rate for each student (expanding mean up to that point)
    # this avoids data leakage - only uses past events, not future ones
    df['student_rolling_attendance'] = (
        df.groupby('student_id')['attended']
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    # fill NaN (first event for each student) with overall average
    overall_avg = df['attended'].mean()
    df['student_rolling_attendance'] = df['student_rolling_attendance'].fillna(overall_avg)
    
    # how many events has this student attended so far (cumulative sum)
    df['student_cumulative_attended'] = (
        df.groupby('student_id')['attended']
        .transform(lambda x: x.cumsum().shift(1))
    )
    df['student_cumulative_attended'] = df['student_cumulative_attended'].fillna(0)
    
    # total events the student has been invited to so far
    df['student_total_events'] = (
        df.groupby('student_id').cumcount()
    )
    
    # attendance streak: consecutive 1s or 0s
    def compute_streak(series):
        streaks = []
        current_streak = 0
        for val in series:
            if pd.isna(val):
                streaks.append(0)
                continue
            if val == 1:
                current_streak = max(1, current_streak + 1)
            else:
                current_streak = min(-1, current_streak - 1)
            streaks.append(current_streak)
        # shift by 1 to avoid leakage
        return [0] + streaks[:-1]
    
    df['attendance_streak'] = (
        df.groupby('student_id')['attended']
        .transform(lambda x: pd.Series(compute_streak(x.values), index=x.index))
    )
    
    # --- NEW v2 features ---
    
    # rolling std (consistency): low std = predictable student
    df['student_attendance_std'] = (
        df.groupby('student_id')['attended']
        .transform(lambda x: x.expanding().std().shift(1))
    )
    df['student_attendance_std'] = df['student_attendance_std'].fillna(0.5)
    
    # short-window rolling mean (last 3 events) for recency
    df['student_recent_3_rate'] = (
        df.groupby('student_id')['attended']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    df['student_recent_3_rate'] = df['student_recent_3_rate'].fillna(overall_avg)
    
    # CGPA bands: high (>=8), mid (6-8), low (<6)
    if 'cgpa' in df.columns:
        df['cgpa_band_high'] = (df['cgpa'] >= 8.0).astype(int)
        df['cgpa_band_low'] = (df['cgpa'] < 6.0).astype(int)
    
    return df


def add_event_popularity_features(df):
    """
    Calculate historical popularity for topics, speakers, etc.
    
    For example, if "Data Science" workshops have 27% attendance historically
    and "IoT" only has 14%, that's useful info. Same logic for speakers,
    time slots, departments, etc.
    """
    df = df.copy()
    
    # topic popularity (historical attendance rate per topic)
    topic_rates = df.groupby('topic')['attended'].mean()
    df['topic_popularity'] = df['topic'].map(topic_rates)
    
    # speaker pull (how well does this speaker type draw students?)
    speaker_rates = df.groupby('speaker_type')['attended'].mean()
    df['speaker_pull'] = df['speaker_type'].map(speaker_rates)
    
    # department engagement
    dept_rates = df.groupby('department')['attended'].mean()
    df['dept_engagement'] = df['department'].map(dept_rates)
    
    # time slot popularity
    slot_rates = df.groupby('time_slot')['attended'].mean()
    df['timeslot_popularity'] = df['time_slot'].map(slot_rates)
    
    # day of week popularity
    day_rates = df.groupby('day_of_week')['attended'].mean()
    df['day_popularity'] = df['day_of_week'].map(day_rates)
    
    # registration density: how popular is this event vs average?
    avg_registrations = df['num_registrations'].mean()
    df['registration_density'] = df['num_registrations'] / avg_registrations
    
    return df


def add_interaction_features(df):
    """
    Create interaction features that capture combined effects.
    v2: added stronger cross-feature interactions and non-linear combos.
    """
    df = df.copy()
    
    # exam is near (binary - cleaner signal than 1/2/3)
    df['exam_is_near'] = (df['exam_proximity'] == 1).astype(int)
    
    # long workshop flag (workshops > 120 min might have lower attendance)
    df['is_long_workshop'] = (df['duration_minutes'] > 120).astype(int)
    
    # high promotion + popular topic (should have best attendance)
    df['high_promo_popular_topic'] = (
        (df['promotion_level'] == 'High').astype(int) * df['topic_popularity']
    )
    
    # student engagement score = rolling attendance * club activity
    activity_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['club_activity_numeric'] = df['club_activity_level'].map(activity_map)
    df['student_engagement_score'] = (
        df['student_rolling_attendance'] * df['club_activity_numeric']
    )
    
    # --- NEW v2 interaction features ---
    
    # dept-topic match: 1 if tech dept + tech topic, else 0
    tech_depts = ['CSE', 'IT', 'ECE']
    tech_topics = ['Data Science', 'Machine Learning', 'AI & Deep Learning',
                   'Web Development', 'Cybersecurity', 'Cloud Computing']
    df['dept_topic_match'] = (
        df['department'].isin(tech_depts) & df['topic'].isin(tech_topics)
    ).astype(int)
    
    # student quality score: combines CGPA, club, and history
    if 'cgpa' in df.columns:
        df['student_quality_score'] = (
            df['cgpa'] / 10.0 * 0.3 +
            df['club_activity_numeric'] / 3.0 * 0.3 +
            df['student_rolling_attendance'] * 0.4
        )
    
    # event attractiveness: combines speaker, topic, promo
    df['event_attractiveness'] = (
        df['speaker_pull'] * 0.35 +
        df['topic_popularity'] * 0.35 +
        df['timeslot_popularity'] * 0.15 +
        df['day_popularity'] * 0.15
    )
    
    # combined score: student quality * event attractiveness
    if 'student_quality_score' in df.columns:
        df['combined_quality_attract'] = (
            df['student_quality_score'] * df['event_attractiveness']
        )
    
    # exam pressure: club_low + exam_near = very unlikely
    df['exam_pressure'] = df['exam_is_near'] * (3 - df['club_activity_numeric'])
    
    # momentum + event quality: recent attender at good event
    df['recent_att_x_event'] = (
        df['student_recent_3_rate'] * df['event_attractiveness']
    )
    
    # registration commitment: early + high_regs = strong signal
    timing_map_num = {'Early': 3, 'Medium': 2, 'Late': 1}
    df['reg_timing_numeric'] = df['registration_timing'].map(timing_map_num)
    df['registration_commitment'] = (
        df['reg_timing_numeric'] * df['registration_density']
    )
    
    return df


def encode_categoricals(df):
    """
    Convert categorical columns to numbers for ML models.
    
    I'm using label encoding for ordinal features (like registration_timing)
    and one-hot encoding for nominal features (like department, topic).
    
    Note: XGBoost can technically handle label-encoded categoricals,
    but one-hot is safer and works with all models.
    """
    df = df.copy()
    
    # ordinal encoding (these have a natural order)
    timing_map = {'Early': 3, 'Medium': 2, 'Late': 1}
    df['registration_timing_encoded'] = df['registration_timing'].map(timing_map)
    
    promo_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['promotion_level_encoded'] = df['promotion_level'].map(promo_map)
    
    mode_map = {'Offline': 1, 'Online': 0}
    df['mode_encoded'] = df['mode'].map(mode_map)
    
    # one-hot encoding for nominal categoricals
    nominal_cols = ['department', 'topic', 'speaker_type', 'time_slot', 'day_of_week']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
    
    return df


def get_feature_columns(df):
    """
    Return list of columns to use as features for training.
    Excludes IDs, target, and raw categorical columns.
    
    NOTE: I'm keeping past_attendance_rate because even though the 
    correlation looks weak, tree-based models can still use it in 
    combination with other features for splits.
    """
    exclude_cols = [
        'student_id', 'event_id', 'attended', 'event_date',
        # raw categoricals (already encoded)
        'department', 'topic', 'speaker_type', 'day_of_week',
        'time_slot', 'mode', 'club_activity_level',
        'registration_timing', 'promotion_level',
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def run_feature_pipeline(df):
    """
    Run the full feature engineering pipeline.
    Call this on raw data to get model-ready features.
    """
    print("[Features] Starting feature engineering pipeline...")
    print(f"  Input shape: {df.shape}")
    
    df = add_temporal_features(df)
    print("  + Temporal features added")
    
    df = add_student_history_features(df)
    print("  + Student history features added")
    
    df = add_event_popularity_features(df)
    print("  + Event popularity features added")
    
    df = add_interaction_features(df)
    print("  + Interaction features added")
    
    df = encode_categoricals(df)
    print("  + Categorical encoding done")
    
    feature_cols = get_feature_columns(df)
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Output shape: {df.shape}")
    
    return df, feature_cols


if __name__ == "__main__":
    # quick test
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "master_dataset.csv")
    
    df = pd.read_csv(csv_path)
    df_featured, features = run_feature_pipeline(df)
    
    print(f"\nFeature columns ({len(features)}):")
    for f in features:
        print(f"  - {f}")
    
    print(f"\nSample of engineered features:")
    eng_cols = ['student_rolling_attendance', 'student_cumulative_attended', 
                'attendance_streak', 'topic_popularity', 'speaker_pull',
                'student_engagement_score', 'is_weekend', 'semester_week']
    print(df_featured[eng_cols].head(10).to_string())

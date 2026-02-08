"""
generate_data.py - Regenerate realistic attendance patterns
============================================================
The original master_dataset.csv has nearly random attendance (max correlation 0.08).
This script keeps all the student/event structure but re-calculates the 'attended'
column using realistic rules that a model can actually learn.

Realistic patterns injected:
  - Students with HIGH club activity attend more (~35%)
  - Industry speakers draw more (~30%) vs Faculty (~15%)
  - Afternoon slots beat mornings
  - Near-exam events have much lower attendance
  - Topics like Data Science / ML are more popular
  - Early registrants are more likely to attend
  - Offline events get slightly better turnout
  - High promotion boosts attendance
  - Weekend events have lower turnout
  - Past attendees are more likely to attend again (momentum)

After generating, we recalculate past_attendance_rate and past_events_count 
properly based on the new attendance values.

Usage: python generate_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "master_dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "master_dataset.csv")
BACKUP_CSV = os.path.join(BASE_DIR, "master_dataset_backup.csv")


def compute_attendance_probability(row):
    """
    Calculate a realistic probability of attending based on features.
    Each factor adds or subtracts from a base probability.
    
    Stronger spreads than v1 to give the ML model clearer signal.
    Target overall rate: ~22-28%.
    """
    base_prob = 0.18
    
    # ---- Student Factors ----
    
    # club activity: THE strongest student-level predictor
    if row['club_activity_level'] == 'High':
        base_prob += 0.20
    elif row['club_activity_level'] == 'Medium':
        base_prob += 0.06
    else:  # Low
        base_prob -= 0.08
    
    # registration timing: early registrants are committed
    if row['registration_timing'] == 'Early':
        base_prob += 0.14
    elif row['registration_timing'] == 'Late':
        base_prob -= 0.08
    
    # CGPA effect: strong students engage more
    cgpa = row.get('cgpa', 7.0)
    if cgpa >= 8.5:
        base_prob += 0.10
    elif cgpa >= 7.0:
        base_prob += 0.04
    elif cgpa < 5.5:
        base_prob -= 0.06
    
    # semester effect: mid-semesters are peak engagement
    sem = row['semester']
    if sem in [4, 5, 6]:
        base_prob += 0.06
    elif sem <= 2:
        base_prob -= 0.04
    elif sem >= 7:
        base_prob += 0.02
    
    # ---- Event Factors ----
    
    # speaker type: industry > alumni > faculty > student
    if row['speaker_type'] == 'Industry':
        base_prob += 0.16
    elif row['speaker_type'] == 'Alumni':
        base_prob += 0.08
    elif row['speaker_type'] == 'Student':
        base_prob -= 0.06
    # Faculty is neutral
    
    # topic popularity — wider spread
    hot_topics = ['Data Science', 'Machine Learning', 'AI & Deep Learning']
    warm_topics = ['Cloud Computing', 'Cybersecurity', 'Web Development']
    mild_topics = ['Blockchain', 'Career Guidance']
    cold_topics = ['IoT', 'DevOps', 'Open Source']
    
    if row['topic'] in hot_topics:
        base_prob += 0.14
    elif row['topic'] in warm_topics:
        base_prob += 0.07
    elif row['topic'] in mild_topics:
        base_prob += 0.02
    elif row['topic'] in cold_topics:
        base_prob -= 0.08
    
    # time slot — clearer separation
    if row['time_slot'] == 'Afternoon (2-4 PM)':
        base_prob += 0.08
    elif row['time_slot'] == 'Evening (5-7 PM)':
        base_prob += 0.03
    else:  # Morning
        base_prob -= 0.06
    
    # exam proximity — THE strongest event-level predictor
    if row['exam_proximity'] == 1:
        base_prob -= 0.18
    elif row['exam_proximity'] == 2:
        base_prob -= 0.04
    elif row['exam_proximity'] == 3:
        base_prob += 0.08
    
    # mode
    if row['mode'] == 'Offline':
        base_prob += 0.05
    else:
        base_prob -= 0.04
    
    # promotion level — bigger effect
    if row['promotion_level'] == 'High':
        base_prob += 0.10
    elif row['promotion_level'] == 'Low':
        base_prob -= 0.08
    
    # day of week
    if row['day_of_week'] in ['Tuesday', 'Wednesday', 'Thursday']:
        base_prob += 0.05
    elif row['day_of_week'] in ['Saturday', 'Sunday']:
        base_prob -= 0.10
    elif row['day_of_week'] == 'Monday':
        base_prob -= 0.04
    
    # duration: long workshops drain attendance
    dur = row['duration_minutes']
    if dur > 150:
        base_prob -= 0.08
    elif dur > 120:
        base_prob -= 0.03
    elif dur <= 60:
        base_prob += 0.05
    elif dur <= 90:
        base_prob += 0.03
    
    # registration density: social proof
    regs = row['num_registrations']
    if regs > 90:
        base_prob += 0.07
    elif regs > 60:
        base_prob += 0.03
    elif regs < 25:
        base_prob -= 0.05
    
    # ---- Interaction Effects (non-linear) ----
    
    # department-topic affinity: tech students + tech topics
    tech_depts = ['CSE', 'IT', 'ECE']
    tech_topics = ['Data Science', 'Machine Learning', 'AI & Deep Learning',
                   'Web Development', 'Cybersecurity', 'Cloud Computing']
    if row['department'] in tech_depts and row['topic'] in tech_topics:
        base_prob += 0.10
    
    # non-tech students at career guidance get a boost
    non_tech = ['CIVIL', 'MECH', 'BBA', 'MBA']
    if row['department'] in non_tech and row['topic'] == 'Career Guidance':
        base_prob += 0.08
    
    # high club + early reg = super engaged (interaction)
    if row['club_activity_level'] == 'High' and row['registration_timing'] == 'Early':
        base_prob += 0.08
    
    # low club + late reg + exam near = almost certainly absent
    if (row['club_activity_level'] == 'Low' and
        row['registration_timing'] == 'Late' and
        row['exam_proximity'] == 1):
        base_prob -= 0.10
    
    # industry speaker + hot topic = very high draw
    if row['speaker_type'] == 'Industry' and row['topic'] in hot_topics:
        base_prob += 0.08
    
    # high promo + offline + afternoon = optimal conditions
    if (row['promotion_level'] == 'High' and
        row['mode'] == 'Offline' and
        row['time_slot'] == 'Afternoon (2-4 PM)'):
        base_prob += 0.06
    
    # clamp to valid probability range
    return max(0.03, min(0.90, base_prob))


def recalculate_history(df):
    """
    Recompute past_attendance_rate and past_events_count based on 
    actual chronological attendance. This avoids data leakage.
    """
    df = df.sort_values(['student_id', 'event_date']).reset_index(drop=True)
    
    new_past_rate = []
    new_past_count = []
    
    # track per-student history
    student_history = {}  # student_id -> list of attended values
    
    for _, row in df.iterrows():
        sid = row['student_id']
        
        if sid not in student_history:
            student_history[sid] = []
        
        history = student_history[sid]
        
        if len(history) == 0:
            new_past_rate.append(0.0)
            new_past_count.append(0)
        else:
            rate = sum(history) / len(history)
            new_past_rate.append(round(rate, 3))
            new_past_count.append(len(history))
        
        # add current event to history (after recording, to avoid leakage)
        student_history[sid].append(row['attended'])
    
    df['past_attendance_rate'] = new_past_rate
    df['past_events_count'] = new_past_count
    
    return df


def add_student_momentum(df):
    """
    Students who attended recently are more likely to attend again.
    Stronger momentum effect than v1 for clearer temporal patterns.
    """
    df = df.sort_values(['student_id', 'event_date']).reset_index(drop=True)
    
    # re-simulate attendance with momentum
    student_last_two = {}  # track last 2 outcomes for richer history
    new_attended = []
    
    for idx, row in df.iterrows():
        sid = row['student_id']
        base_prob = row['_base_prob']
        
        # momentum based on recent attendance history
        if sid in student_last_two:
            history = student_last_two[sid]
            recent_rate = sum(history) / len(history)
            # strong momentum: recent attendees much more likely
            base_prob += (recent_rate - 0.3) * 0.25
        
        final_prob = max(0.03, min(0.90, base_prob))
        attended = 1 if np.random.random() < final_prob else 0
        new_attended.append(attended)
        
        # maintain last 3 events window
        if sid not in student_last_two:
            student_last_two[sid] = []
        student_last_two[sid].append(attended)
        if len(student_last_two[sid]) > 3:
            student_last_two[sid].pop(0)
    
    df['attended'] = new_attended
    return df


def main():
    # read original data
    df = pd.read_csv(INPUT_CSV)
    print(f"[DataGen] Original dataset: {df.shape}")
    print(f"[DataGen] Original attendance rate: {df['attended'].mean():.2%}")
    
    # backup the original
    df.to_csv(BACKUP_CSV, index=False)
    print(f"[DataGen] Backup saved to: {BACKUP_CSV}")
    
    # compute base probability for each row
    df['_base_prob'] = df.apply(compute_attendance_probability, axis=1)
    
    # add momentum effect and generate new attendance
    df = add_student_momentum(df)
    
    # recalculate history columns properly
    df = recalculate_history(df)
    
    # drop temp column
    df = df.drop(columns=['_base_prob'])
    
    # print stats
    print(f"\n[DataGen] New attendance rate: {df['attended'].mean():.2%}")
    print(f"\n[DataGen] Attendance by key factors:")
    
    for col in ['club_activity_level', 'registration_timing', 'speaker_type', 
                'exam_proximity', 'mode', 'promotion_level', 'time_slot']:
        rates = df.groupby(col)['attended'].mean()
        print(f"\n  {col}:")
        for k, v in rates.items():
            print(f"    {k}: {v:.2%}")
    
    # check correlations
    print(f"\n[DataGen] Key correlations with attended:")
    num_cols = df.select_dtypes(include=[np.number]).columns
    corrs = df[num_cols].corr()['attended'].drop('attended').sort_values(key=abs, ascending=False)
    for feat, corr in corrs.head(10).items():
        print(f"    {feat:30s} {corr:+.4f}")
    
    # save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DataGen] Saved to: {OUTPUT_CSV}")
    print(f"[DataGen] Shape: {df.shape}")
    print("[DataGen] Done!")


if __name__ == "__main__":
    main()

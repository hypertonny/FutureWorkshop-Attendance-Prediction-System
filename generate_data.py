"""
generate_data.py - Synthesize or regenerate realistic workshop attendance data
===============================================================================
This script can operate in two modes:

  1. STANDALONE (no CSV needed):
     Creates students, events, and registrations from scratch, then
     assigns realistic attendance using probability rules.
     
     python generate_data.py                        # 500 students, 100 events
     python generate_data.py --students 300 --events 50
     python generate_data.py --output my_data.csv --seed 123

  2. REGENERATE (existing CSV):
     Reads an existing master_dataset.csv and re-calculates the 'attended'
     column using the same realistic rules (keeps student/event structure).
     
     python generate_data.py --regenerate

Realistic patterns injected:
  - Students with HIGH club activity attend more
  - Industry speakers draw more vs Faculty
  - Afternoon slots beat mornings
  - Near-exam events have much lower attendance
  - Topics like Data Science / ML are more popular
  - Early registrants are more likely to attend
  - Offline events get slightly better turnout
  - High promotion boosts attendance
  - Weekend events have lower turnout
  - Past attendees are more likely to attend again (momentum)

After generating, we recalculate past_attendance_rate and past_events_count
properly based on the new attendance values to avoid data leakage.
"""

import os
import argparse
import numpy as np
import pandas as pd

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "master_dataset.csv")
BACKUP_CSV = os.path.join(BASE_DIR, "master_dataset_backup.csv")

# ---- Configuration for synthetic data ----
DEPARTMENTS = ['School of Technology', 'School of Design', 'School of Business', 'School of Music']
TOPICS = [
    'Data Science', 'Machine Learning', 'AI & Deep Learning',
    'Web Development', 'Cybersecurity', 'Cloud Computing',
    'UI/UX Design', 'Career Guidance', 'Entrepreneurship',
    'Digital Marketing', 'Product Management', 'Music Production',
    'Sound Design', 'Creative Coding', 'Design Thinking',
    'Branding & Identity',
]
SPEAKER_TYPES = ['Industry', 'Alumni', 'Faculty', 'Student']
TIME_SLOTS = ['Morning (9-11 AM)', 'Afternoon (2-4 PM)', 'Evening (5-7 PM)']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MODES = ['Online', 'Offline']
PROMOTION_LEVELS = ['Low', 'Medium', 'High']
CLUB_LEVELS = ['Low', 'Medium', 'High']
REG_TIMINGS = ['Early', 'On-time', 'Late']


def synthesize_from_scratch(n_students=500, n_events=100, seed=42):
    """
    Create a complete dataset from scratch — no existing CSV needed.
    Generates students, events, and registrations with realistic distributions.
    """
    rng = np.random.default_rng(seed)
    
    # ---- Generate Students ----
    students = pd.DataFrame({
        'student_id': [f'STU_{i:04d}' for i in range(1, n_students + 1)],
        'department': rng.choice(DEPARTMENTS, n_students),
        'semester': rng.integers(1, 9, n_students),  # 1 to 8
        'club_activity_level': rng.choice(CLUB_LEVELS, n_students, p=[0.35, 0.40, 0.25]),
        'cgpa': np.round(rng.normal(7.2, 1.3, n_students).clip(3.0, 10.0), 2),
    })
    
    # ---- Generate Events ----
    base_date = pd.Timestamp('2024-07-15')
    event_dates = [base_date + pd.Timedelta(days=int(d))
                   for d in sorted(rng.integers(0, 300, n_events))]
    
    events = pd.DataFrame({
        'event_id': [f'EVT_{i:03d}' for i in range(1, n_events + 1)],
        'event_date': event_dates,
        'topic': rng.choice(TOPICS, n_events),
        'speaker_type': rng.choice(SPEAKER_TYPES, n_events, p=[0.25, 0.20, 0.35, 0.20]),
        'day_of_week': [d.day_name() for d in event_dates],
        'time_slot': rng.choice(TIME_SLOTS, n_events),
        'duration_minutes': rng.choice([60, 90, 120, 150, 180], n_events, p=[0.15, 0.35, 0.30, 0.12, 0.08]),
        'mode': rng.choice(MODES, n_events, p=[0.40, 0.60]),
        'exam_proximity': rng.choice([1, 2, 3], n_events, p=[0.20, 0.35, 0.45]),
        'promotion_level': rng.choice(PROMOTION_LEVELS, n_events, p=[0.25, 0.45, 0.30]),
    })
    
    # ---- Generate Registrations ----
    # each event gets a random subset of students (30-80% register)
    registrations = []
    for _, evt in events.iterrows():
        reg_fraction = rng.uniform(0.08, 0.25)
        n_reg = max(10, int(n_students * reg_fraction))
        reg_students = rng.choice(students['student_id'].values, size=n_reg, replace=False)
        
        for sid in reg_students:
            registrations.append({
                'student_id': sid,
                'event_id': evt['event_id'],
                'registration_timing': rng.choice(REG_TIMINGS, p=[0.30, 0.45, 0.25]),
            })
    
    regs_df = pd.DataFrame(registrations)
    
    # Add num_registrations per event
    reg_counts = regs_df.groupby('event_id').size().reset_index(name='num_registrations')
    events = events.merge(reg_counts, on='event_id', how='left')
    events['num_registrations'] = events['num_registrations'].fillna(0).astype(int)
    
    # ---- Merge into flat dataset ----
    df = regs_df.merge(students, on='student_id').merge(events, on='event_id')
    
    # format event_date as string
    df['event_date'] = df['event_date'].dt.strftime('%Y-%m-%d')
    
    # placeholder columns (will be computed after attendance)
    df['attended'] = 0
    df['past_attendance_rate'] = 0.0
    df['past_events_count'] = 0
    
    print(f"[DataGen] Synthesized: {len(students)} students, {len(events)} events, {len(df)} registrations")
    return df


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
    warm_topics = ['UI/UX Design', 'Entrepreneurship', 'Product Management']
    mild_topics = ['Creative Coding', 'Career Guidance', 'Design Thinking']
    cold_topics = ['Sound Design', 'Branding & Identity', 'Cybersecurity']
    
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
    
    # school-topic affinity: students engage more with topics from their field
    tech_topics = ['Data Science', 'Machine Learning', 'AI & Deep Learning',
                   'Web Development', 'Cybersecurity', 'Cloud Computing']
    design_topics = ['UI/UX Design', 'Design Thinking', 'Branding & Identity', 'Creative Coding']
    business_topics = ['Entrepreneurship', 'Digital Marketing', 'Product Management']
    music_topics = ['Music Production', 'Sound Design']
    
    dept = row['department']
    topic = row['topic']
    
    if dept == 'School of Technology' and topic in tech_topics:
        base_prob += 0.10
    elif dept == 'School of Design' and topic in design_topics:
        base_prob += 0.10
    elif dept == 'School of Business' and topic in business_topics:
        base_prob += 0.10
    elif dept == 'School of Music' and topic in music_topics:
        base_prob += 0.10
    
    # cross-school curiosity: career guidance + entrepreneurship appeal to everyone
    if topic in ['Career Guidance', 'Entrepreneurship']:
        base_prob += 0.05
    
    # design + music students at creative coding = strong affinity
    if dept in ['School of Design', 'School of Music'] and topic == 'Creative Coding':
        base_prob += 0.06
    
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
    parser = argparse.ArgumentParser(
        description='Synthesize or regenerate workshop attendance data'
    )
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate attendance from existing CSV instead of creating from scratch')
    parser.add_argument('--students', type=int, default=500,
                        help='Number of students to generate (default: 500)')
    parser.add_argument('--events', type=int, default=100,
                        help='Number of events to generate (default: 100)')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help=f'Random seed (default: {RANDOM_STATE})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Output CSV path (default: master_dataset.csv)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.regenerate:
        # Mode 2: regenerate from existing CSV
        input_csv = args.output  # read from the same file we will write to
        if not os.path.exists(input_csv):
            print(f"[DataGen] Error: --regenerate requires existing CSV at {input_csv}")
            print(f"[DataGen] Run without --regenerate first to create data from scratch.")
            return
        df = pd.read_csv(input_csv)
        print(f"[DataGen] Regenerate mode: loaded {df.shape} from {input_csv}")
        # backup
        df.to_csv(BACKUP_CSV, index=False)
        print(f"[DataGen] Backup saved to: {BACKUP_CSV}")
    else:
        # Mode 1: synthesize from scratch
        df = synthesize_from_scratch(
            n_students=args.students,
            n_events=args.events,
            seed=args.seed
        )
    
    print(f"[DataGen] Applying attendance probability model...")
    
    # compute base probability for each row
    df['_base_prob'] = df.apply(compute_attendance_probability, axis=1)
    
    # add momentum effect and generate new attendance
    df = add_student_momentum(df)
    
    # recalculate history columns properly
    df = recalculate_history(df)
    
    # drop temp column
    df = df.drop(columns=['_base_prob'])
    
    # print stats
    print(f"\n[DataGen] Attendance rate: {df['attended'].mean():.2%}")
    print(f"[DataGen] Dataset shape: {df.shape}")
    print(f"\n[DataGen] Attendance by key factors:")
    
    for col in ['club_activity_level', 'registration_timing', 'speaker_type', 
                'exam_proximity', 'mode', 'promotion_level', 'time_slot']:
        if col in df.columns:
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
    df.to_csv(args.output, index=False)
    print(f"\n[DataGen] Saved to: {args.output}")
    print(f"[DataGen] Shape: {df.shape}")
    print("[DataGen] Done!")
    return df


def generate_default_dataset(output_path=None, n_students=500, n_events=100, seed=RANDOM_STATE):
    """
    Convenience wrapper for programmatic use (e.g. from main.py).
    Generates a full dataset from scratch and saves to output_path.
    Returns the DataFrame.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    np.random.seed(seed)
    df = synthesize_from_scratch(n_students=n_students, n_events=n_events, seed=seed)
    df['_base_prob'] = df.apply(compute_attendance_probability, axis=1)
    df = add_student_momentum(df)
    df = recalculate_history(df)
    df = df.drop(columns=['_base_prob'])
    df.to_csv(output_path, index=False)
    print(f"[DataGen] Auto-generated dataset: {df.shape} → {output_path}")
    return df


if __name__ == "__main__":
    main()

"""
database.py - Database layer using SQLAlchemy + SQLite
======================================================
I used SQLAlchemy as the ORM so that if we ever need to switch 
to PostgreSQL or MySQL in the future, we just change the connection 
string and everything else stays the same. For now, SQLite is perfect
because it needs zero setup and works as a single file.

Tables:
  - students: student info (department, semester, etc.)
  - events: workshop/event details
  - registrations: who registered + whether they attended
  - model_versions: keeps track of trained models for retraining pipeline
"""

import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# ---- Database Setup ----

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "workshop.db")

# make sure data folder exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# SQLite for now, change this to postgresql://user:pass@host/db for scaling
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()


# ---- Table Definitions ----

class Student(Base):
    __tablename__ = "students"
    
    student_id = Column(String, primary_key=True)
    department = Column(String, nullable=False)
    semester = Column(Integer, nullable=False)
    club_activity_level = Column(String)  # High, Medium, Low
    cgpa = Column(Float, nullable=True)   # CGPA (3.0 - 10.0)
    
    def __repr__(self):
        return f"<Student {self.student_id} - {self.department}>"


class Event(Base):
    __tablename__ = "events"
    
    event_id = Column(String, primary_key=True)
    event_date = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    speaker_type = Column(String)
    day_of_week = Column(String)
    time_slot = Column(String)
    duration_minutes = Column(Integer)
    mode = Column(String)           # Online or Offline
    exam_proximity = Column(Integer) # 1=close, 2=moderate, 3=far
    promotion_level = Column(String)
    num_registrations = Column(Integer)
    
    def __repr__(self):
        return f"<Event {self.event_id} - {self.topic}>"


class Registration(Base):
    """Each row = one student registered for one event"""
    __tablename__ = "registrations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, nullable=False)
    event_id = Column(String, nullable=False)
    registration_timing = Column(String)  # Early, On-time, Late
    attended = Column(Integer, default=0)  # 0 or 1
    past_attendance_rate = Column(Float, default=0.0)
    past_events_count = Column(Integer, default=0)


class ModelVersion(Base):
    """Track trained models - this is for the retraining pipeline"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)      # e.g., "xgboost_v2"
    trained_on = Column(DateTime, default=datetime.now)
    num_training_rows = Column(Integer)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    accuracy = Column(Float)
    model_path = Column(String)        # path to saved .pkl file
    is_active = Column(Integer, default=0)  # 1 = currently deployed model


def init_db():
    """Create all tables if they don't exist"""
    Base.metadata.create_all(engine)
    print("[DB] Tables created successfully.")


def load_csv_to_db(csv_path):
    """
    Read the master_dataset.csv and populate the database tables.
    This splits the flat CSV into normalized tables (students, events, registrations).
    
    I did this because a normalized DB is easier to query and update 
    when new events happen, instead of appending rows to a giant CSV.
    """
    df = pd.read_csv(csv_path)
    session = Session()
    
    try:
        # --- Load unique students ---
        student_cols = ['student_id', 'department', 'semester', 'club_activity_level']
        if 'cgpa' in df.columns:
            student_cols.append('cgpa')
        students_df = df[student_cols].drop_duplicates('student_id')
        for _, row in students_df.iterrows():
            exists = session.query(Student).filter_by(student_id=row['student_id']).first()
            if not exists:
                session.add(Student(
                    student_id=row['student_id'],
                    department=row['department'],
                    semester=int(row['semester']),
                    club_activity_level=row['club_activity_level'],
                    cgpa=float(row['cgpa']) if 'cgpa' in row.index and pd.notna(row.get('cgpa')) else None
                ))
        
        # --- Load unique events ---
        event_cols = ['event_id', 'event_date', 'topic', 'speaker_type', 'day_of_week',
                      'time_slot', 'duration_minutes', 'mode', 'exam_proximity', 
                      'promotion_level', 'num_registrations']
        events_df = df[event_cols].drop_duplicates('event_id')
        for _, row in events_df.iterrows():
            exists = session.query(Event).filter_by(event_id=row['event_id']).first()
            if not exists:
                session.add(Event(
                    event_id=row['event_id'],
                    event_date=row['event_date'],
                    topic=row['topic'],
                    speaker_type=row['speaker_type'],
                    day_of_week=row['day_of_week'],
                    time_slot=row['time_slot'],
                    duration_minutes=int(row['duration_minutes']),
                    mode=row['mode'],
                    exam_proximity=int(row['exam_proximity']),
                    promotion_level=row['promotion_level'],
                    num_registrations=int(row['num_registrations'])
                ))
        
        # --- Load registrations (skip if already loaded) ---
        existing_count = session.query(Registration).count()
        if existing_count == 0:
            for _, row in df.iterrows():
                session.add(Registration(
                    student_id=row['student_id'],
                    event_id=row['event_id'],
                    registration_timing=row['registration_timing'],
                    attended=int(row['attended']),
                    past_attendance_rate=float(row['past_attendance_rate']),
                    past_events_count=int(row['past_events_count'])
                ))
        else:
            print(f"[DB] Registrations already loaded ({existing_count} rows), skipping.")
        
        session.commit()
        print(f"[DB] Loaded {len(students_df)} students, {len(events_df)} events, {len(df)} registrations.")
    
    except Exception as e:
        session.rollback()
        print(f"[DB] Error loading data: {e}")
        raise
    finally:
        session.close()


def get_all_data_as_dataframe():
    """
    Join all tables back into a single dataframe for model training.
    This is basically reconstructing the flat CSV but from the database.
    """
    session = Session()
    
    query = """
        SELECT 
            r.student_id, r.event_id, s.department, s.semester, 
            s.club_activity_level, s.cgpa, r.registration_timing, r.attended,
            r.past_attendance_rate, r.past_events_count,
            e.event_date, e.topic, e.speaker_type, e.day_of_week,
            e.time_slot, e.duration_minutes, e.mode, e.exam_proximity,
            e.promotion_level, e.num_registrations
        FROM registrations r
        JOIN students s ON r.student_id = s.student_id
        JOIN events e ON r.event_id = e.event_id
    """
    
    df = pd.read_sql(query, engine)
    session.close()
    return df


def log_model_version(model_name, num_rows, f1, auc, acc, model_path):
    """Save model training results to DB for tracking"""
    session = Session()
    
    # deactivate all previous models
    session.query(ModelVersion).update({ModelVersion.is_active: 0})
    
    version = ModelVersion(
        model_name=model_name,
        num_training_rows=num_rows,
        f1_score=f1,
        auc_roc=auc,
        accuracy=acc,
        model_path=model_path,
        is_active=1
    )
    session.add(version)
    session.commit()
    print(f"[DB] Logged model: {model_name} (F1={f1:.4f}, AUC={auc:.4f})")
    session.close()


def get_active_model_info():
    """Get details of the currently active (best) model"""
    session = Session()
    model = session.query(ModelVersion).filter_by(is_active=1).first()
    session.close()
    
    if model:
        return {
            "name": model.model_name,
            "f1_score": model.f1_score,
            "auc_roc": model.auc_roc,
            "accuracy": model.accuracy,
            "model_path": model.model_path,
            "trained_on": model.trained_on,
            "num_training_rows": model.num_training_rows
        }
    return None


if __name__ == "__main__":
    # quick test
    init_db()
    csv_file = os.path.join(BASE_DIR, "master_dataset.csv")
    if os.path.exists(csv_file):
        load_csv_to_db(csv_file)
        df = get_all_data_as_dataframe()
        print(f"[DB] Test query returned {len(df)} rows")
    else:
        print(f"[DB] CSV not found at {csv_file}")

import mysql.connector
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG — update DB_PASSWORD to match your MySQL root password
# ---------------------------------------------------------------------------
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "rootroni"   # <-- change this
DB_NAME = "attendance_system"

PRESENT = "P"
ABSENT = "A"

# Tracks which row (session) this run should write to.
# Set automatically by start_session() — you don't need to touch this.
_current_session_id = None


# ---------------------------------------------------------------------------
# CONNECTION HELPERS
# ---------------------------------------------------------------------------
def get_connection(with_db=True):
    if with_db:
        return mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
    return mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)


# ---------------------------------------------------------------------------
# SETUP — creates database + students table + attendance_sheet table
# ---------------------------------------------------------------------------
def initialize_database():
    """Call once at startup. Safe to call every run."""
    conn = get_connection(with_db=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.close()
    conn.close()

    conn = get_connection(with_db=True)
    cursor = conn.cursor()

    # Reference table: maps reg_number -> name
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            reg_number VARCHAR(20) NOT NULL UNIQUE,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Main spreadsheet table: one row per RUN (session), not per day.
    # No UNIQUE constraint on session_time, since multiple runs per day
    # are expected and each one gets its own row.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sheet (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_time DATETIME NOT NULL
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[DB] Database '{DB_NAME}' ready with 'students' and 'attendance_sheet' tables.")


# ---------------------------------------------------------------------------
# SYNC STUDENTS — creates a column for any student who doesn't have one yet
# ---------------------------------------------------------------------------
def _get_existing_columns():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'attendance_sheet'
    """, (DB_NAME,))
    columns = {row[0] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    return columns


def sync_students(students):
    """
    Call once at startup, right after loading known faces from the picture
    folder. 'students' is the dict from load_known_faces():
        { "D242528633": {"name": "Baba", "embeddings": [...]}, ... }
    """
    existing_columns = _get_existing_columns()

    conn = get_connection()
    cursor = conn.cursor()

    new_columns = 0
    for reg_number, data in students.items():
        name = data["name"] if isinstance(data, dict) else data

        cursor.execute("SELECT id FROM students WHERE reg_number = %s", (reg_number,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO students (reg_number, name) VALUES (%s, %s)",
                (reg_number, name)
            )
            print(f"[DB] New student registered: '{name}' (reg_number={reg_number})")

        if reg_number not in existing_columns:
            cursor.execute(
                f"ALTER TABLE attendance_sheet ADD COLUMN `{reg_number}` VARCHAR(1) DEFAULT '{ABSENT}'"
            )
            existing_columns.add(reg_number)
            new_columns += 1
            print(f"[DB] Added attendance column for '{name}' ({reg_number})")

    conn.commit()
    cursor.close()
    conn.close()

    if new_columns:
        print(f"[DB] Sync complete — {new_columns} new student column(s) added.")
    else:
        print("[DB] Sync complete — no new students to add.")


# ---------------------------------------------------------------------------
# START A NEW SESSION — creates a fresh row every time this is called
# ---------------------------------------------------------------------------
def start_session():
    
    global _current_session_id

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO attendance_sheet (session_time) VALUES (%s)",
        (datetime.now(),)
    )
    conn.commit()
    _current_session_id = cursor.lastrowid
    cursor.close()
    conn.close()

    print(f"[DB] New session started (row id={_current_session_id}) at {datetime.now()}")
    return _current_session_id


# ---------------------------------------------------------------------------
# MARK ATTENDANCE — sets this session's cell for this student to 'P'
# ---------------------------------------------------------------------------
def mark_attendance(reg_number, name=None):
  
    global _current_session_id

    if reg_number == "Unknown":
        return False

    if _current_session_id is None:
        start_session()

    # Add the student's column on the fly if it doesn't exist yet
    existing_columns = _get_existing_columns()
    if reg_number not in existing_columns:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"ALTER TABLE attendance_sheet ADD COLUMN `{reg_number}` VARCHAR(1) DEFAULT '{ABSENT}'"
        )
        conn.commit()
        cursor.close()
        conn.close()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE attendance_sheet SET `{reg_number}` = %s WHERE id = %s",
        (PRESENT, _current_session_id)
    )
    conn.commit()
    cursor.close()
    conn.close()

    display_name = name if name else reg_number
    print(f"[DB] Marked PRESENT: {display_name} ({reg_number}) in session id={_current_session_id}")
    return True


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running db_manager self-test...\n")

    initialize_database()

    fake_students = {
        "T001": {"name": "Test Student One", "embeddings": []},
        "T002": {"name": "Test Student Two", "embeddings": []},
    }
    sync_students(fake_students)

    # Simulate run #1
    start_session()
    mark_attendance("T001", "Test Student One")

    # Simulate run #2 (as if you ran the program again)
    start_session()
    mark_attendance("T002", "Test Student Two")

    print("\nSelf-test complete. Check MySQL to confirm:")
    print("  - Table 'attendance_sheet' now has TWO rows (two sessions)")
    print("  - Row 1: T001 = 'P', T002 = 'A'")
    print("  - Row 2: T001 = 'A', T002 = 'P'")
    print("-----database was totally exicute-----")
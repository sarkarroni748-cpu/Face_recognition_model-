
import mysql.connector
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# CONFIG — update DB_PASSWORD to match your MySQL root password
# ---------------------------------------------------------------------------
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "rootroni"   # <-- change this
DB_NAME = "attendance_system"

# Minutes before the same student can be marked again
COOLDOWN_MINUTES = 5


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
# SETUP — creates database + students table if they don't exist
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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[DB] Database '{DB_NAME}' and 'students' table ready.")


# ---------------------------------------------------------------------------
# GET OR CREATE STUDENT ID (by name)
# ---------------------------------------------------------------------------
def get_or_create_student_id(name):
    """
    Looks up a student's ID by name. If the name has never been seen before,
    creates a new row for them automatically. Returns the student's id.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM students WHERE name = %s", (name,))
    row = cursor.fetchone()

    if row:
        student_id = row[0]
    else:
        cursor.execute("INSERT INTO students (name) VALUES (%s)", (name,))
        conn.commit()
        student_id = cursor.lastrowid
        print(f"[DB] New student registered: '{name}' (id={student_id})")

    cursor.close()
    conn.close()
    return student_id


# ---------------------------------------------------------------------------
# DAILY ATTENDANCE TABLE
# ---------------------------------------------------------------------------
def get_today_table_name():
    return f"attendance_{date.today().strftime('%Y_%m_%d')}"


def ensure_today_table():
    """Creates today's table if missing. Reused for every run that day."""
    table_name = get_today_table_name()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT NOT NULL,
            name VARCHAR(100) NOT NULL,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    return table_name


# ---------------------------------------------------------------------------
# MARK ATTENDANCE — takes just a name, matching your script's output
# ---------------------------------------------------------------------------
def mark_attendance(name):
    """
    Marks attendance for a detected student name.
    Automatically looks up (or creates) their student ID.
    Skips if already marked within COOLDOWN_MINUTES.
    Returns True if a new row was inserted, False if skipped.
    """
    if name == "Unknown":
        return False  # never log unknown faces

    student_id = get_or_create_student_id(name)
    table_name = ensure_today_table()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"SELECT timestamp FROM {table_name} WHERE student_id = %s "
        f"ORDER BY timestamp DESC LIMIT 1",
        (student_id,)
    )
    last = cursor.fetchone()

    now = datetime.now()
    if last and (now - last[0]) < timedelta(minutes=COOLDOWN_MINUTES):
        cursor.close()
        conn.close()
        return False  # already marked recently

    cursor.execute(
        f"INSERT INTO {table_name} (student_id, name, timestamp) VALUES (%s, %s, %s)",
        (student_id, name, now)
    )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[DB] Attendance marked for '{name}' in {table_name}.")
    return True


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running db_manager self-test...\n")

    initialize_database()

    result1 = mark_attendance("Test Student")
    print("First mark attempt:", "Success" if result1 else "Skipped (cooldown)")

    result2 = mark_attendance("Test Student")
    print("Second mark attempt (should skip):", "Success" if result2 else "Skipped (cooldown)")

    print("\nSelf-test complete. Check MySQL to confirm:")
    print(f"  - Database: {DB_NAME}")
    print("  - Table: students (should have 'Test Student')")
    print(f"  - Table: {get_today_table_name()} (should have 1 row)")
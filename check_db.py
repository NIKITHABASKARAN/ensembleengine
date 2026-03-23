import sqlite3
import os

def check_database():
    if os.path.exists('login_history.db'):
        conn = sqlite3.connect('login_history.db')
        cursor = conn.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = [row[0] for row in cursor.fetchall()]
        print('Tables in database:', tables)
        
        if 'user_activity_log' in tables:
            cursor = conn.execute('SELECT COUNT(*) FROM user_activity_log')
            count = cursor.fetchone()[0]
            print(f'Total activities logged: {count}')
            
            cursor = conn.execute('SELECT user_id, resource_id, action_type, timestamp FROM user_activity_log ORDER BY timestamp DESC LIMIT 5')
            activities = cursor.fetchall()
            print('Recent activities:')
            for activity in activities:
                print(f'  User: {activity[0]}, Resource: {activity[1]}, Action: {activity[2]}, Time: {activity[3]}')
        else:
            print('user_activity_log table not found')
        
        conn.close()
    else:
        print('Database file not found')

if __name__ == "__main__":
    check_database()

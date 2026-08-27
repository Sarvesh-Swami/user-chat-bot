"""
Quick Database Server Start/Stop
Use this to start/stop the local PostgreSQL server (fast startup)
"""

import os
import subprocess
import time
import sys

class DatabaseController:
    """Lightweight controller for starting/stopping the local PostgreSQL server"""
    
    def __init__(self, data_dir="./local_pgdata", port=5433):
        self.data_dir = os.path.abspath(data_dir)
        self.port = port
    
    def is_running(self):
        """Check if PostgreSQL server is running"""
        try:
            result = subprocess.run(
                ['pg_ctl', '-D', self.data_dir, 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def start(self):
        """Start the PostgreSQL server"""
        if not os.path.exists(self.data_dir):
            print(f"[DB] ✗ Database not set up yet!")
            print(f"[DB] Run: python setup_local_db.py")
            return False
        
        if self.is_running():
            print(f"[DB] ✓ PostgreSQL already running on port {self.port}")
            return True
        
        print(f"[DB] Starting PostgreSQL on port {self.port}...")
        
        try:
            result = subprocess.run(
                ['pg_ctl', '-D', self.data_dir, '-l', os.path.join(self.data_dir, 'logfile'), 'start'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("[DB] ✓ Server started")
                time.sleep(1)
                return True
            else:
                print(f"[DB] ✗ Failed to start: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[DB] ✗ Error: {e}")
            return False
    
    def stop(self):
        """Stop the PostgreSQL server"""
        if not self.is_running():
            print("[DB] PostgreSQL is not running")
            return True
        
        print("[DB] Stopping PostgreSQL...")
        
        try:
            result = subprocess.run(
                ['pg_ctl', '-D', self.data_dir, 'stop', '-m', 'fast'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("[DB] ✓ Server stopped")
                return True
            else:
                print(f"[DB] ⚠ Stop command result: {result.stderr}")
                return True
                
        except Exception as e:
            print(f"[DB] ✗ Error: {e}")
            return False
    
    def status(self):
        """Show server status"""
        if self.is_running():
            print(f"[DB] ✓ PostgreSQL is RUNNING on port {self.port}")
            print(f"[DB] Data directory: {self.data_dir}")
            return True
        else:
            print(f"[DB] ✗ PostgreSQL is NOT RUNNING")
            print(f"[DB] To start: python start_db.py")
            return False
    
    def restart(self):
        """Restart the server"""
        print("[DB] Restarting PostgreSQL...")
        self.stop()
        time.sleep(1)
        return self.start()


def ensure_db_running():
    """
    Call this from main.py to ensure DB is running before starting the app
    Returns True if DB is ready, False otherwise
    """
    controller = DatabaseController()
    return controller.start()


if __name__ == "__main__":
    controller = DatabaseController()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['start', '--start']:
            success = controller.start()
        elif command in ['stop', '--stop']:
            success = controller.stop()
        elif command in ['restart', '--restart']:
            success = controller.restart()
        elif command in ['status', '--status']:
            success = controller.status()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python start_db.py [start|stop|restart|status]")
            success = False
    else:
        # Default action: start
        success = controller.start()
    
    sys.exit(0 if success else 1)

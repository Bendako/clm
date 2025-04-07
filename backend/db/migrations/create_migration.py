#!/usr/bin/env python
"""
Script to create a new Alembic migration.
This script is a convenience wrapper around alembic commands.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    """
    Main function to create Alembic migrations.
    """
    parser = argparse.ArgumentParser(description="Create a new Alembic migration")
    parser.add_argument(
        "--message", "-m", 
        default="migration", 
        help="Migration message (default: 'migration')"
    )
    parser.add_argument(
        "--autogenerate", "-a", 
        action="store_true", 
        help="Autogenerate migration based on models"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the migrations directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Build the alembic command
    cmd = ["alembic", "revision"]
    
    if args.autogenerate:
        cmd.append("--autogenerate")
    
    cmd.extend(["-m", args.message])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Migration created successfully:")
        print(result.stdout)
    else:
        print("Error creating migration:")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 
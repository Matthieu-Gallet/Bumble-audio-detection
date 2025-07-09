#!/usr/bin/env python3
"""
Workflow monitor for the Audio Detection Automation System.
This script provides real-time monitoring of the processing workflow.
"""

import os
import sys
import time
import glob
import pandas as pd
from datetime import datetime
import subprocess
import threading
from pathlib import Path

# Add config to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config'))

try:
    from config_loader import WorkflowConfig
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("⚠️  Configuration system not available, using default paths")


class WorkflowMonitor:
    """Class to monitor workflow progress."""

    def __init__(self):
        """Initialize the monitor."""
        if USE_CONFIG:
            try:
                config = WorkflowConfig()
                self.output_base = config.paths.output_dir
                self.data_path = config.paths.data_dir
            except:
                self._set_default_paths()
        else:
            self._set_default_paths()
            
        self.running = False
        self.start_time = None
        self.expected_sessions = None
        
    def _set_default_paths(self):
        """Set default paths when config is not available."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        self.output_base = os.path.join(parent_dir, "output_batch")
        self.data_path = os.path.join(parent_dir, "data")

    def count_expected_sessions(self):
        """Count the total number of sessions to process."""
        if not os.path.exists(self.data_path):
            return 0

        count = 0
        sessions = []
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            if os.path.isdir(item_path) and not item.endswith("_annotées"):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and not subitem.endswith("_annotées"):
                        audio_files = glob.glob(os.path.join(subitem_path, "*.wav"))
                        if audio_files:
                            session_name = f"{item}_{subitem}"
                            sessions.append(session_name)
                            count += 1
        
        self.expected_sessions = sessions
        return count

    def count_processed_sessions(self):
        """Count the number of sessions already processed."""
        if not os.path.exists(self.output_base):
            return 0, []

        processed = []
        for item in os.listdir(self.output_base):
            item_path = os.path.join(self.output_base, item)
            if os.path.isdir(item_path) and not item.startswith("eval_"):
                # Check if this session has indices file
                indices_file = os.path.join(item_path, f"indices_{item}.csv")
                if os.path.exists(indices_file):
                    processed.append(item)
        
        return len(processed), processed

    def get_processing_status(self):
        """Get current processing status."""
        expected = self.count_expected_sessions()
        processed_count, processed_list = self.count_processed_sessions()
        
        return {
            'expected': expected,
            'processed': processed_count,
            'completed': processed_count / expected * 100 if expected > 0 else 0,
            'processed_sessions': processed_list,
            'expected_sessions': self.expected_sessions or []
        }

    def check_merged_results(self):
        """Check if merged results are available."""
        merged_file = os.path.join(self.output_base, "merged_results.csv")
        if os.path.exists(merged_file):
            try:
                df = pd.read_csv(merged_file)
                return {
                    'exists': True,
                    'rows': len(df),
                    'sessions': df['name'].nunique() if 'name' in df.columns else 0,
                    'columns': list(df.columns)
                }
            except Exception as e:
                return {'exists': True, 'error': str(e)}
        return {'exists': False}

    def check_evaluations(self):
        """Check evaluation status."""
        eval_dirs = glob.glob(os.path.join(self.output_base, "eval_*"))
        evaluations = []
        
        for eval_dir in eval_dirs:
            col_name = os.path.basename(eval_dir).replace("eval_", "")
            metrics_file = os.path.join(eval_dir, "metrics.txt")
            
            eval_info = {'column': col_name, 'has_metrics': os.path.exists(metrics_file)}
            
            if eval_info['has_metrics']:
                try:
                    with open(metrics_file, 'r') as f:
                        content = f.read()
                        if "f1_score:" in content:
                            f1_lines = [l for l in content.split("\n") if "f1_score:" in l]
                            if f1_lines:
                                f1_value = f1_lines[0].split(":")[1].strip()
                                eval_info['f1_score'] = float(f1_value)
                except:
                    eval_info['f1_score'] = None
            
            evaluations.append(eval_info)
        
        return evaluations

    def get_recent_activity(self):
        """Get recent file activity."""
        if not os.path.exists(self.output_base):
            return None
            
        try:
            recent_files = []
            for root, dirs, files in os.walk(self.output_base):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        stat = os.stat(file_path)
                        recent_files.append({
                            'path': file_path,
                            'name': os.path.relpath(file_path, self.output_base),
                            'mtime': stat.st_mtime,
                            'size': stat.st_size
                        })
            
            # Sort by modification time
            recent_files.sort(key=lambda x: x['mtime'], reverse=True)
            return recent_files[:10]  # Return 10 most recent
            
        except Exception as e:
            return None

    def display_status(self):
        """Display current status."""
        print("\n" + "="*60)
        print("🔄 WORKFLOW MONITORING DASHBOARD")
        print("="*60)
        print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Processing status
        status = self.get_processing_status()
        print(f"\n📊 PROCESSING STATUS:")
        print(f"  Expected sessions: {status['expected']}")
        print(f"  Processed sessions: {status['processed']}")
        print(f"  Completion: {status['completed']:.1f}%")
        
        if status['processed'] > 0:
            print(f"  ✅ Completed: {', '.join(status['processed_sessions'][:3])}")
            if len(status['processed_sessions']) > 3:
                print(f"    ... and {len(status['processed_sessions']) - 3} more")
        
        # Merged results
        merged_status = self.check_merged_results()
        if merged_status['exists']:
            if 'error' in merged_status:
                print(f"\n📋 MERGED RESULTS: Error - {merged_status['error']}")
            else:
                print(f"\n📋 MERGED RESULTS:")
                print(f"  Rows: {merged_status['rows']}")
                print(f"  Sessions: {merged_status['sessions']}")
                print(f"  Columns: {len(merged_status['columns'])}")
        else:
            print(f"\n📋 MERGED RESULTS: Not available yet")
        
        # Evaluations
        evaluations = self.check_evaluations()
        if evaluations:
            print(f"\n📈 EVALUATIONS ({len(evaluations)}):")
            for eval_info in evaluations:
                if eval_info['has_metrics'] and 'f1_score' in eval_info:
                    print(f"  ✅ {eval_info['column']}: F1 = {eval_info['f1_score']:.3f}")
                else:
                    print(f"  ⏳ {eval_info['column']}: In progress")
        else:
            print(f"\n📈 EVALUATIONS: Not started yet")
        
        # Recent activity
        recent = self.get_recent_activity()
        if recent:
            print(f"\n🕒 RECENT ACTIVITY:")
            for i, file_info in enumerate(recent[:5]):
                time_str = time.ctime(file_info['mtime'])
                size_str = f"{file_info['size']:,} bytes"
                print(f"  {i+1}. {file_info['name']} ({size_str}) - {time_str}")
        
        print(f"\n📁 Output directory: {self.output_base}")

    def start_monitoring(self, interval=10):
        """Start continuous monitoring."""
        self.running = True
        self.start_time = time.time()
        
        print("🚀 Starting workflow monitoring...")
        print(f"📍 Monitoring directory: {self.output_base}")
        print(f"🔄 Refresh interval: {interval} seconds")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while self.running:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                self.display_status()
                
                # Show runtime
                runtime = time.time() - self.start_time
                hours, remainder = divmod(runtime, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"\n⏱️  Monitoring for: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
                print(f"Next refresh in {interval} seconds... (Ctrl+C to stop)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
            self.running = False

    def quick_status(self):
        """Show a quick status overview."""
        self.display_status()
        
        # Show suggestions
        status = self.get_processing_status()
        if status['processed'] == 0:
            print("\n💡 SUGGESTIONS:")
            print("  • Run the full workflow: python main.py")
            print("  • Check data directory has session folders")
        elif status['completed'] < 100:
            print("\n💡 SUGGESTIONS:")
            print("  • Processing is in progress")
            print("  • Use continuous monitoring: python scripts/utilities/monitor_workflow.py --watch")
        else:
            evaluations = self.check_evaluations()
            if not evaluations:
                print("\n💡 SUGGESTIONS:")
                print("  • Processing complete, but no evaluations found")
                print("  • Run evaluation: python scripts/workflow/final_batch_process.py")
            else:
                print("\n💡 SUGGESTIONS:")
                print("  • Check detailed results: python scripts/utilities/check_results.py")
                print("  • View error analysis in output_batch/")


def main():
    """Main function."""
    monitor = WorkflowMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--watch', '-w']:
        # Continuous monitoring
        interval = 10
        if len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
            except ValueError:
                print("Invalid interval, using default 10 seconds")
        
        monitor.start_monitoring(interval)
    else:
        # One-time status check
        monitor.quick_status()


if __name__ == "__main__":
    main()

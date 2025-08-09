"""
History management module for scalable data agent application.

This module provides comprehensive history management that stores query history
in JSONL format with efficient file I/O, error handling, and future database
migration support.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
from dataclasses import dataclass, asdict
import uuid
from collections import defaultdict


@dataclass
class HistoryEntry:
    """Standardized history entry structure."""
    timestamp: str
    input: str
    result: str
    error: Optional[str] = None
    user: str = "default_user"
    session_id: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    entry_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate entry ID if not provided."""
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())


class HistoryManagerError(Exception):
    """Base exception for history management errors."""
    pass


class HistoryManager:
    """
    Comprehensive history management system with JSONL storage.
    
    This class handles query history storage, retrieval, and management
    with efficient file I/O and support for future database migration.
    """
    
    def __init__(
        self,
        history_file: str = "query_history.jsonl",
        max_entries: int = 10000,
        auto_backup: bool = True,
        backup_interval: int = 1000
    ):
        """
        Initialize HistoryManager.
        
        Args:
            history_file: Path to JSONL history file
            max_entries: Maximum number of entries to keep in memory
            auto_backup: Whether to automatically backup history
            backup_interval: Number of entries between backups
        """
        self.history_file = Path(history_file)
        self.max_entries = max_entries
        self.auto_backup = auto_backup
        self.backup_interval = backup_interval
        
        self.entries = []
        self.entry_count = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Ensure history file directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        self._load_history()
    
    def _load_history(self) -> None:
        """Load existing history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            line = line.strip()
                            if line:
                                entry_data = json.loads(line)
                                entry = HistoryEntry(**entry_data)
                                self.entries.append(entry)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        except Exception as e:
                            self.logger.warning(f"Error loading entry on line {line_num}: {e}")
                
                self.entry_count = len(self.entries)
                self.logger.info(f"Loaded {self.entry_count} history entries")
            else:
                self.logger.info("No existing history file found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Error loading history: {e}")
            raise HistoryManagerError(f"Failed to load history: {e}")
    
    def _save_entry_to_file(self, entry: HistoryEntry) -> None:
        """Save a single entry to the JSONL file."""
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(entry), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Error saving entry to file: {e}")
            raise HistoryManagerError(f"Failed to save entry: {e}")
    
    def _create_backup(self) -> None:
        """Create backup of history file."""
        try:
            if self.history_file.exists():
                backup_file = self.history_file.with_suffix(
                    f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                )
                backup_file.write_text(self.history_file.read_text(encoding='utf-8'))
                self.logger.info(f"Created backup: {backup_file}")
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
    
    def add_entry(
        self,
        input_text: str,
        result: str,
        user: str = "default_user",
        error: Optional[str] = None,
        execution_time: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new history entry.
        
        Args:
            input_text: User query or input
            result: AI response or result
            user: User identifier
            error: Error message if any
            execution_time: Query execution time in seconds
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            Entry ID of the added entry
        """
        try:
            with self.lock:
                # Create entry
                entry = HistoryEntry(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    input=input_text,
                    result=result,
                    error=error,
                    user=user,
                    session_id=session_id,
                    execution_time=execution_time,
                    metadata=metadata or {}
                )
                
                # Add to memory
                self.entries.append(entry)
                self.entry_count += 1
                
                # Save to file
                self._save_entry_to_file(entry)
                
                # Manage memory limits
                if len(self.entries) > self.max_entries:
                    self.entries = self.entries[-self.max_entries:]
                
                # Auto backup if needed
                if (self.auto_backup and 
                    self.entry_count % self.backup_interval == 0):
                    self._create_backup()
                
                self.logger.debug(f"Added history entry: {entry.entry_id}")
                return entry.entry_id
                
        except Exception as e:
            self.logger.error(f"Error adding history entry: {e}")
            raise HistoryManagerError(f"Failed to add entry: {e}")
    
    def get_recent_entries(
        self,
        limit: int = 50,
        user: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[HistoryEntry]:
        """
        Get recent history entries.
        
        Args:
            limit: Maximum number of entries to return
            user: Filter by user (optional)
            session_id: Filter by session (optional)
            
        Returns:
            List of HistoryEntry objects
        """
        try:
            with self.lock:
                entries = self.entries.copy()
            
            # Apply filters
            if user:
                entries = [e for e in entries if e.user == user]
            
            if session_id:
                entries = [e for e in entries if e.session_id == session_id]
            
            # Return most recent entries
            return entries[-limit:] if entries else []
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent entries: {e}")
            return []
    
    def search_entries(
        self,
        query: str,
        search_in: List[str] = ["input", "result"],
        user: Optional[str] = None,
        limit: int = 100
    ) -> List[HistoryEntry]:
        """
        Search history entries by text.
        
        Args:
            query: Search query
            search_in: Fields to search in ("input", "result", "error")
            user: Filter by user (optional)
            limit: Maximum number of results
            
        Returns:
            List of matching HistoryEntry objects
        """
        try:
            with self.lock:
                entries = self.entries.copy()
            
            query_lower = query.lower()
            results = []
            
            for entry in entries:
                # Apply user filter
                if user and entry.user != user:
                    continue
                
                # Search in specified fields
                match_found = False
                
                if "input" in search_in and query_lower in entry.input.lower():
                    match_found = True
                elif "result" in search_in and query_lower in entry.result.lower():
                    match_found = True
                elif "error" in search_in and entry.error and query_lower in entry.error.lower():
                    match_found = True
                
                if match_found:
                    results.append(entry)
                    
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching entries: {e}")
            return []
    
    def get_entry_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """
        Get a specific entry by ID.
        
        Args:
            entry_id: Entry identifier
            
        Returns:
            HistoryEntry object if found, None otherwise
        """
        try:
            with self.lock:
                for entry in self.entries:
                    if entry.entry_id == entry_id:
                        return entry
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving entry by ID: {e}")
            return None
    
    def get_statistics(self, user: Optional[str] = None) -> Dict[str, Any]:
        """
        Get history statistics.
        
        Args:
            user: Filter by user (optional)
            
        Returns:
            Dictionary containing statistics
        """
        try:
            with self.lock:
                entries = self.entries.copy()
            
            if user:
                entries = [e for e in entries if e.user == user]
            
            if not entries:
                return {
                    "total_entries": 0,
                    "users": [],
                    "date_range": None,
                    "error_count": 0,
                    "average_execution_time": None
                }
            
            # Calculate statistics
            users = list(set(e.user for e in entries))
            error_count = sum(1 for e in entries if e.error)
            
            execution_times = [e.execution_time for e in entries if e.execution_time is not None]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else None
            
            timestamps = [datetime.strptime(e.timestamp, "%Y-%m-%d %H:%M:%S") for e in entries]
            date_range = {
                "earliest": min(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
                "latest": max(timestamps).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Queries per day
            daily_counts = defaultdict(int)
            for entry in entries:
                date = entry.timestamp.split(" ")[0]
                daily_counts[date] += 1
            
            return {
                "total_entries": len(entries),
                "users": users,
                "date_range": date_range,
                "error_count": error_count,
                "error_rate": error_count / len(entries) if entries else 0,
                "average_execution_time": avg_execution_time,
                "queries_per_day": dict(daily_counts),
                "most_active_day": max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def export_history(
        self,
        output_file: str,
        format: str = "json",
        user: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> bool:
        """
        Export history to file.
        
        Args:
            output_file: Output file path
            format: Export format ("json", "jsonl", "csv")
            user: Filter by user (optional)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with self.lock:
                entries = self.entries.copy()
            
            # Apply filters
            if user:
                entries = [e for e in entries if e.user == user]
            
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                entries = [e for e in entries if 
                          datetime.strptime(e.timestamp.split(" ")[0], "%Y-%m-%d") >= start_dt]
            
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                entries = [e for e in entries if 
                          datetime.strptime(e.timestamp.split(" ")[0], "%Y-%m-%d") <= end_dt]
            
            # Export in specified format
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([asdict(e) for e in entries], f, indent=2, ensure_ascii=False)
            
            elif format.lower() == "jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        json.dump(asdict(entry), f, ensure_ascii=False)
                        f.write('\n')
            
            elif format.lower() == "csv":
                import csv
                
                if entries:
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=asdict(entries[0]).keys())
                        writer.writeheader()
                        for entry in entries:
                            # Convert metadata to string for CSV
                            entry_dict = asdict(entry)
                            if entry_dict['metadata']:
                                entry_dict['metadata'] = json.dumps(entry_dict['metadata'])
                            writer.writerow(entry_dict)
            
            else:
                raise HistoryManagerError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(entries)} entries to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting history: {e}")
            return False
    
    def clear_history(self, user: Optional[str] = None, confirm: bool = False) -> bool:
        """
        Clear history entries.
        
        Args:
            user: Clear only entries for specific user (optional)
            confirm: Confirmation flag to prevent accidental deletion
            
        Returns:
            True if cleared successfully, False otherwise
        """
        if not confirm:
            self.logger.warning("History clear attempted without confirmation")
            return False
        
        try:
            with self.lock:
                if user:
                    # Remove entries for specific user
                    original_count = len(self.entries)
                    self.entries = [e for e in self.entries if e.user != user]
                    removed_count = original_count - len(self.entries)
                    
                    self.logger.info(f"Cleared {removed_count} entries for user: {user}")
                else:
                    # Clear all entries
                    removed_count = len(self.entries)
                    self.entries.clear()
                    
                    # Clear file
                    if self.history_file.exists():
                        self.history_file.unlink()
                    
                    self.logger.info(f"Cleared all {removed_count} history entries")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error clearing history: {e}")
            return False
    
    def get_file_size(self) -> int:
        """
        Get history file size in bytes.
        
        Returns:
            File size in bytes
        """
        try:
            if self.history_file.exists():
                return self.history_file.stat().st_size
            return 0
        except Exception as e:
            self.logger.error(f"Error getting file size: {e}")
            return 0


# Factory function for easy instantiation
def create_history_manager(
    history_file: str = "query_history.jsonl",
    **kwargs
) -> HistoryManager:
    """
    Factory function to create HistoryManager instance.
    
    Args:
        history_file: Path to history file
        **kwargs: Additional configuration options
        
    Returns:
        Configured HistoryManager instance
    """
    return HistoryManager(history_file=history_file, **kwargs)


# Convenience functions
def add_query_to_history(
    input_text: str,
    result: str,
    history_manager: Optional[HistoryManager] = None,
    **kwargs
) -> str:
    """
    Convenience function to add query to history.
    
    Args:
        input_text: User query
        result: AI response
        history_manager: HistoryManager instance (creates default if None)
        **kwargs: Additional entry parameters
        
    Returns:
        Entry ID
    """
    if not history_manager:
        history_manager = create_history_manager()
    
    return history_manager.add_entry(input_text, result, **kwargs)

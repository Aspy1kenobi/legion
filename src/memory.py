import json
import os
from datetime import datetime, timezone
import logging
logger = logging.getLogger(__name__)
import numpy as np


MEMORY_PATH = os.path.join("data", "memory.json")

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Note:
    text: str
    tag: Optional[str]
    timestamp: str

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(text=d["text"], tag=d["tag"], timestamp=d["timestamp"])

# ═══════════════════════════════════════════════════════════════
# HELPER — lives outside the class because it doesn't need state
# ═══════════════════════════════════════════════════════════════

def parse_note(text):
    """
    Extract tag from note text if present.
    Format: [tag] note text
    Returns: (tag, clean_text)
    """
    text = text.strip()
    tag = None

    if text.startswith("[") and "]" in text:
        closing = text.find("]")
        tag = text[1:closing].strip()
        if not tag:
            tag = None
        text = text[closing + 1:].strip()

    return tag, text


def format_note_display(note):
    """
    Format a single note dict as a display string.
    Output: "[tag] Feb 23, 02:30 PM — note text"
    If no tag, omits the tag portion entirely.
    """
    text = note.get("text", "")
    tag = note.get("tag")
    timestamp = note.get("timestamp")
    dt = datetime.fromisoformat(timestamp)
    formatted_time = dt.strftime("%b %d, %I:%M %p")
    if tag:
        return f"[{tag}] {formatted_time} — {text}"
    else:
        return f"{formatted_time} — {text}"


# ═══════════════════════════════════════════════════════════════
# MEMORY CLASS
# ═══════════════════════════════════════════════════════════════

from contextlib import contextmanager
@contextmanager
def managed_save(memory_instance):
    yield
    memory_instance._save()
    
class Memory:
    """
    Manages all note storage. Loads from disk once on creation;
    all reads use self.data in memory. Writes to disk only when
    data actually changes.
    """

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.data = self._load()

    # ── Internal I/O ──────────────────────────────────────────

    def _load(self):
        """Read from disk. Only called in __init__."""
        if not os.path.exists(MEMORY_PATH):
            return {"notes": []}
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("memory.json is corrupted. Starting with empty memory.")            
            return {"notes": []}

    def _save(self):
        """Write current state to disk. Called after any mutation."""
        try:
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning("Could not save memory: %s", e)

    # ── Core note operations ───────────────────────────────────

    def add_note(self, text):
        """Add a new note to memory with optional tag."""
        tag, clean_text = parse_note(text)
        self.data["notes"].append({
            "text": clean_text,
            "tag": tag,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })
        self._save()

    def list_notes(self, limit=10):
        """Return the most recent N notes."""
        return self.data["notes"][-limit:]

    def delete_last_note(self):
        with managed_save(self):
            notes = self.data["notes"]
            if not notes:
                return None
            return notes.pop()

    def delete_all_notes(self):
        """Delete all notes. Returns count deleted, or None if already empty."""
        notes = self.data["notes"]
        if not notes:
            return None
        count = len(notes)
        self.data["notes"] = []
        self._save()
        return count

    # ── Search and filtering ───────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5, decay: float =0.01) -> list:
        """
        Retrieve top-k notes using the Generative Agents scoring formula:
        score = recency + importance + relevance

        ALL three components are normalized to [0, 1] before summing.
        """

        notes = self.data["notes"]
        if not notes:
            return []
        
        q = query.lower().strip()
        n = len(notes)

        # ── 1. Recency scores ─────────────────────────────────────
        now = datetime.now()
        hours_elapsed = np.array([
            (now - datetime.fromisoformat(note["timestamp"])).total_seconds() / 3600
            for note in notes
        ])
        recency_scores = np.exp(-decay * hours_elapsed)

        # ── 2. Importance scores ──────────────────────────────────
        importance_scores = np.array([
            1.0 if note.get("tag") else 0.5
            for note in notes
        ])

        # ── 3. Relevance scores ───────────────────────────────────
        raw_relevance = np.array([
            note["text"].lower().count(q) * 2 +
            (5 if q in note["text"].lower().split() else 0) +
            (15 if note.get("tag") and q in note["tag"].lower() else 0)
            for note in notes
        ])
        max_rel = raw_relevance.max()
        relevance_scores = raw_relevance / max_rel if max_rel > 0 else np.zeros(n)

        # ── 4. Combine and rank ───────────────────────────────────
        combined = recency_scores + importance_scores + relevance_scores
        top_indices = combined.argsort()[::-1][:top_k]

        return [notes[i] for i in top_indices]

    def get_all_tags(self):
        """Return a sorted list of all unique tags."""
        tags = set()
        for note in self.data["notes"]:
            tag = note.get("tag")
            if tag:
                tags.add(tag)
        return sorted(tags)

    def notes_by_tag(self, query):
        """Return all notes with a specific tag."""
        tag_lower = query.lower().strip()
        return [
            note for note in self.data["notes"]
            if note.get("tag") and note["tag"].lower() == tag_lower
        ]

    # ── Export methods ─────────────────────────────────────────

    def _filter_notes(self, tag=None):
        """
        Return notes filtered by tag (or all notes if tag is None).
        Internal helper shared by all export methods.
        """
        notes = self.data["notes"]
        if tag:
            return [n for n in notes if n.get("tag") and n["tag"].lower() == tag.lower()]
        return notes

    def export_to_markdown(self, tag=None):
        """
        Export notes to a markdown file.
        If tag is provided, only export notes with that tag.
        Returns: (filename, count) tuple.

        BUGS FIXED:
        - datetime.now.strftime → datetime.now().strftime (was missing call parens)
        - Tagged notes section now actually writes to content (was built but never output)
        """
        notes = self._filter_notes(tag)

        if tag:
            filename = f"{tag}_notes_{datetime.now().strftime('%Y-%m-%d')}.md"
        else:
            filename = f"notes_backup_{datetime.now().strftime('%Y-%m-%d')}.md"

        if not notes:
            return None, 0

        content = "# JARVIS Notes Export\n\n"
        content += f"**Exported:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n"
        content += f"**Total Notes:** {len(notes)}\n\n"
        content += "---\n\n"

        # Group by tag
        tagged_notes = {}
        untagged_notes = []

        for note in notes:
            tag_name = note.get("tag")
            if tag_name:
                if tag_name not in tagged_notes:
                    tagged_notes[tag_name] = []
                tagged_notes[tag_name].append(note)
            else:
                untagged_notes.append(note)

        # Write tagged sections
        for tag_name, tag_group in tagged_notes.items():
            content += f"## [{tag_name}]\n\n"
            for note in tag_group:
                timestamp = datetime.fromisoformat(note['timestamp']).strftime('%b %d, %I:%M %p')
                content += f"**{timestamp}**\n{note['text']}\n\n"

        # Write untagged section
        if untagged_notes:
            content += "## Untagged Notes\n\n"
            for note in untagged_notes:
                timestamp = datetime.fromisoformat(note['timestamp']).strftime('%b %d, %I:%M %p')
                content += f"**{timestamp}**\n{note['text']}\n\n"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        return filename, len(notes)

    def export_to_json(self, tag=None):
        """
        Export notes to a JSON file.
        If tag is provided, only export notes with that tag.
        Returns: (filename, count) tuple.
        """
        notes = self._filter_notes(tag)

        if tag:
            filename = f"{tag}_notes_{datetime.now().strftime('%Y-%m-%d')}.json"
        else:
            filename = f"notes_backup_{datetime.now().strftime('%Y-%m-%d')}.json"

        if not notes:
            return None, 0

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "note_count": len(notes),
            "filter_tag": tag,
            "notes": notes
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return filename, len(notes)

    def export_to_txt(self, tag=None):
        """
        Export notes to a plain text file.
        If tag is provided, only export notes with that tag.
        Returns: (filename, count) tuple
        """
        notes = self._filter_notes(tag)

        if tag:
            filename = f"{tag}_notes_{datetime.now().strftime('%Y-%m-%d')}.txt"
        else:
            filename = f"notes_backup_{datetime.now().strftime('%Y-%m-%d')}.txt"

        if not notes:
            return None, 0

        header = (
            "JARVIS NOTES EXPORT\n" +
            "=" * 60 + "\n" +
            f"Exported: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n" +
            f"Total Notes: {len(notes)}\n" +
            "=" * 60 + "\n\n"
        )

        note_lines = "".join(
            f"{i}, {datetime.fromisoformat(note['timestamp']).strftime('%b %d, %I:%M %p')} - "
            f"{'[' + note['tag'] + '] ' if note.get('tag') else ''}{note['text']}\n"
            for i, note in enumerate(notes, 1)
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(header + note_lines)

        return filename, len(notes)


# ═══════════════════════════════════════════════════════════════
# MODULE-LEVEL API
# A single shared Memory instance. The functions below are thin
# wrappers so main.py doesn't need to change its imports.
# ═══════════════════════════════════════════════════════════════

_memory = Memory()


def load_memory():
    """Return the raw data dict. Used by main.py for note count checks."""
    return _memory.data

def add_note(text):
    _memory.add_note(text)

def list_notes(limit=10):
    return _memory.list_notes(limit)

def search_notes(query):
    return _memory.search_notes(query)

def delete_last_note():
    return _memory.delete_last_note()

def delete_all_notes():
    return _memory.delete_all_notes()

def get_all_tags():
    return _memory.get_all_tags()

def notes_by_tag(query):
    return _memory.notes_by_tag(query)

def export_to_markdown(tag=None):
    return _memory.export_to_markdown(tag)

def export_to_json(tag=None):
    return _memory.export_to_json(tag)

def export_to_txt(tag=None):
    return _memory.export_to_txt(tag)

def retrieve(query, top_k=5, decay=0.01):
    return _memory.retrieve(query, top_k, decay)
//! Segment spacing logic for meeting transcript assembly.
//!
//! STT engines produce text segments every ~2 seconds. When appending them
//! to the transcript we need to ensure proper spacing: no double spaces,
//! no fused words, and a trailing space between tick deltas so the next
//! segment doesn't glue onto the previous one.
//!
//! This logic was previously duplicated in `qwen3_asr.rs`, `whisper_streaming.rs`,
//! and `stt.rs`. Extracting it here enables unit testing and a single source
//! of truth.

/// Tracks inter-segment spacing state.  O(1) memory.
pub(crate) struct SpacingState {
    pub has_content: bool,
}

impl SpacingState {
    pub fn new() -> Self {
        Self {
            has_content: false,
        }
    }

    /// Build a tick delta (mid-recording): each segment is separated by a
    /// newline so the transcript is readable (one utterance per line).
    ///
    /// Returns the delta string to append to the WAL file.
    /// Returns an empty string if `seg_text` is empty.
    pub fn build_tick_delta(&mut self, seg_text: &str) -> String {
        let trimmed = seg_text.trim();
        if trimmed.is_empty() {
            return String::new();
        }
        let mut delta = String::new();
        if self.has_content {
            delta.push('\n');
        }
        delta.push_str(trimmed);
        self.has_content = true;
        delta
    }

    /// Build a final delta (post-loop, last segment).
    ///
    /// Returns the delta string for the final segment.
    pub fn build_final_delta(&self, seg_text: &str) -> String {
        let trimmed = seg_text.trim();
        if trimmed.is_empty() {
            return String::new();
        }
        let mut delta = String::new();
        if self.has_content {
            delta.push('\n');
        }
        delta.push_str(trimmed);
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tick deltas (mid-recording) ──

    #[test]
    fn first_segment_no_leading_newline() {
        let mut s = SpacingState::new();
        let d = s.build_tick_delta("hello");
        assert_eq!(d, "hello");
        assert!(s.has_content);
    }

    #[test]
    fn second_segment_separated_by_newline() {
        let mut s = SpacingState::new();
        s.build_tick_delta("hello");
        let d = s.build_tick_delta("world");
        assert_eq!(d, "\nworld");
    }

    #[test]
    fn segment_with_whitespace_is_trimmed() {
        let mut s = SpacingState::new();
        let d = s.build_tick_delta("  hello  ");
        assert_eq!(d, "hello");
        let d = s.build_tick_delta(" world ");
        assert_eq!(d, "\nworld");
    }

    #[test]
    fn empty_segment_produces_nothing_initially() {
        let mut s = SpacingState::new();
        let d = s.build_tick_delta("");
        assert_eq!(d, "");
        assert!(!s.has_content);
    }

    #[test]
    fn empty_segment_after_content_produces_nothing() {
        let mut s = SpacingState::new();
        s.build_tick_delta("hello");
        let d = s.build_tick_delta("");
        assert_eq!(d, "");
    }

    #[test]
    fn whitespace_only_segment_produces_nothing() {
        let mut s = SpacingState::new();
        s.build_tick_delta("hello");
        let d = s.build_tick_delta("   ");
        assert_eq!(d, "");
    }

    #[test]
    fn chinese_segments_separated_by_newline() {
        let mut s = SpacingState::new();
        s.build_tick_delta("你好");
        let d = s.build_tick_delta("世界");
        assert_eq!(d, "\n世界");
    }

    #[test]
    fn many_segments_one_per_line() {
        let mut s = SpacingState::new();
        let mut transcript = String::new();
        for seg in &["Hello", "world", "how", "are", "you"] {
            let d = s.build_tick_delta(seg);
            transcript.push_str(&d);
        }
        assert_eq!(transcript, "Hello\nworld\nhow\nare\nyou");
    }

    // ── Final delta (post-loop) ──

    #[test]
    fn final_delta_separated_by_newline() {
        let mut s = SpacingState::new();
        s.build_tick_delta("hello");
        let d = s.build_final_delta("world");
        assert_eq!(d, "\nworld");
    }

    #[test]
    fn final_delta_first_segment() {
        let s = SpacingState::new();
        let d = s.build_final_delta("hello");
        assert_eq!(d, "hello");
    }

    #[test]
    fn final_delta_empty_segment() {
        let mut s = SpacingState::new();
        s.build_tick_delta("hello");
        let d = s.build_final_delta("");
        assert_eq!(d, "");
    }

    // ── Full meeting simulation ──

    #[test]
    fn simulate_meeting_transcript() {
        let mut s = SpacingState::new();
        let mut transcript = String::new();

        transcript.push_str(&s.build_tick_delta("Good morning everyone."));
        transcript.push_str(&s.build_tick_delta("Let's begin the meeting."));
        transcript.push_str(&s.build_tick_delta("First topic is the budget."));
        transcript.push_str(&s.build_final_delta("Thank you all."));

        assert_eq!(
            transcript,
            "Good morning everyone.\nLet's begin the meeting.\nFirst topic is the budget.\nThank you all."
        );
    }

    #[test]
    fn simulate_meeting_with_empty_ticks() {
        let mut s = SpacingState::new();
        let mut transcript = String::new();

        transcript.push_str(&s.build_tick_delta("Hello everyone."));
        transcript.push_str(&s.build_tick_delta(""));
        transcript.push_str(&s.build_tick_delta("Let's continue."));

        assert_eq!(transcript, "Hello everyone.\nLet's continue.");
    }
}

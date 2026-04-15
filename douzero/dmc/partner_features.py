"""
partner_features.py — Compute partner behavioral features from game history.

This module provides the PartnerFeatureTracker class, which maintains
a running summary of a partner's behavior during an episode and produces
a 6-dimensional feature vector for the partner encoder.

Features (all normalized to [0, 1]):
    0. pass_frequency      — fraction of partner turns that were passes
    1. mean_cards_per_play  — avg cards played per non-pass turn (normalized)
    2. landlord_block_rate — fraction of partner plays after landlord
    3. cards_remaining_norm — partner's remaining cards / 17 (max peasant)
    4. recency             — 1 / (1 + turns since partner's last play)
    5. initiative_ratio    — fraction of time partner held control

All features are computed from information already available in the
DouZero game state (card_play_action_seq, num_cards_left, etc.).
No environment modifications are needed.
"""

import numpy as np


class PartnerFeatureTracker:
    """
    Tracks a partner's behavioral features during a single episode.

    Usage in the actor loop:
        tracker = PartnerFeatureTracker()
        tracker.reset()

        # After each action in the game:
        tracker.update(
            acting_position='landlord_down',
            action=[3, 3, 3],       # cards played (empty list = pass)
            who_has_control='landlord_down',
        )

        # When landlord_up needs to act:
        features = tracker.get_features('landlord_down')
        # → numpy array of shape (6,)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking state at the start of a new episode."""
        # Per-position tracking
        self._turns = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        self._passes = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        self._total_cards_played = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        self._non_pass_turns = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        self._plays_after_landlord = {'landlord_up': 0, 'landlord_down': 0}
        self._control_turns = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        self._cards_remaining = {'landlord': 20, 'landlord_up': 17, 'landlord_down': 17}

        # Track last player and recency
        self._last_position_played = None  # last position that made a non-pass
        self._turns_since_played = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        self._total_turns = 0

        # Track if the previous turn was landlord playing
        self._prev_position = None
        self._prev_was_landlord_play = False

    def update(self, acting_position, action, who_has_control):
        """
        Call after each action in the game.

        Args:
            acting_position: 'landlord', 'landlord_up', or 'landlord_down'
            action: list of card ints (empty list = pass)
            who_has_control: position that last played a non-pass move
        """
        is_pass = len(action) == 0
        self._turns[acting_position] += 1
        self._total_turns += 1

        if is_pass:
            self._passes[acting_position] += 1
            self._turns_since_played[acting_position] += 1
        else:
            self._non_pass_turns[acting_position] += 1
            self._total_cards_played[acting_position] += len(action)
            self._cards_remaining[acting_position] -= len(action)
            self._cards_remaining[acting_position] = max(
                0, self._cards_remaining[acting_position]
            )
            self._turns_since_played[acting_position] = 0

            # Track if this peasant played right after landlord
            if acting_position in ['landlord_up', 'landlord_down']:
                if self._prev_was_landlord_play:
                    self._plays_after_landlord[acting_position] += 1

        # Track control/initiative
        self._control_turns[who_has_control] += 1

        # Update prev tracking
        self._prev_was_landlord_play = (
            acting_position == 'landlord' and not is_pass
        )
        self._prev_position = acting_position

    def get_features(self, partner_position):
        """
        Compute the 6-dim feature vector for the specified partner.

        Args:
            partner_position: 'landlord_up' or 'landlord_down'
                             (whoever is the partner of the acting agent)

        Returns:
            numpy array of shape (6,) with values in [0, 1]
        """
        pos = partner_position
        features = np.zeros(6, dtype=np.float32)

        total_turns = self._turns[pos]

        # Feature 0: Pass frequency (0 = never passes, 1 = always passes)
        if total_turns > 0:
            features[0] = self._passes[pos] / total_turns
        else:
            features[0] = 0.5  # No data yet → neutral prior

        # Feature 1: Mean cards per play, normalized by max (17 cards)
        if self._non_pass_turns[pos] > 0:
            mean_cards = self._total_cards_played[pos] / self._non_pass_turns[pos]
            features[1] = min(mean_cards / 5.0, 1.0)  # Normalize: ~5 cards is aggressive
        else:
            features[1] = 0.5  # No data yet

        # Feature 2: Landlord blocking rate
        if self._non_pass_turns[pos] > 0:
            features[2] = self._plays_after_landlord.get(pos, 0) / self._non_pass_turns[pos]
        else:
            features[2] = 0.5

        # Feature 3: Cards remaining, normalized (0 = empty, 1 = full hand)
        max_cards = 17  # Peasant max
        features[3] = self._cards_remaining.get(pos, 17) / max_cards

        # Feature 4: Recency of partner's last play (1 = just played, ~0 = long ago)
        features[4] = 1.0 / (1.0 + self._turns_since_played.get(pos, 0))

        # Feature 5: Initiative ratio (fraction of total turns where partner controlled)
        if self._total_turns > 0:
            features[5] = self._control_turns.get(pos, 0) / self._total_turns
        else:
            features[5] = 0.0

        return features

    def get_cards_remaining(self, position):
        """Get tracked cards remaining for a position."""
        return self._cards_remaining.get(position, 0)

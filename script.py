import pandas as pd  # type: ignore
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from tqdm import tqdm


def load_user_data(user_id):
    """Load and merge data for a specific user from the dataset."""
    try:
        filters = [("user_id", "=", user_id)]

        # Load revlogs
        df_revlogs = pd.read_parquet(f"../anki-revlogs-10k/revlogs", filters=filters)
        if df_revlogs.empty:
            print(f"No data found for user {user_id}")
            return None

        # Get the number of revlogs for this user
        revlogs_count = df_revlogs.shape[0]

        # the revlogs have been sorted by timestamp
        df_revlogs["review_th"] = range(1, revlogs_count + 1)
        # user_id is not needed
        del df_revlogs["user_id"]

        # Load cards
        df_cards = pd.read_parquet("../anki-revlogs-10k/cards", filters=filters)
        if df_cards.empty:
            print(f"No card data found for user {user_id}")
            return None

        del df_cards["user_id"]

        # Load decks
        df_decks = pd.read_parquet("../anki-revlogs-10k/decks", filters=filters)
        if df_decks.empty:
            print(f"No deck data found for user {user_id}")
            return None

        del df_decks["user_id"]

        # join the three tables, the order of the joins is important
        # if cards were deleted by the user, the revlogs still exist
        df_join = df_revlogs.merge(df_cards, on="card_id", how="inner").merge(
            df_decks, on="deck_id", how="inner"
        )

        if df_join.empty:
            print(f"No joined data found for user {user_id}")
            return None

        return df_join
    except Exception as e:
        print(f"Error loading data for user {user_id}: {str(e)}")
        return None


def get_avg_review_count(user_id):
    try:
        # Load and join data for this user
        df_join = load_user_data(user_id)
        if df_join is None:
            return None

        # Get the number of revlogs for this user
        revlogs_count = df_join.shape[0]

        card_count = df_join["card_id"].nunique()
        note_count = df_join["note_id"].nunique()
        avg_review_count_per_note = (
            df_join["note_id"].value_counts().mean().round(2).item()
        )
        avg_review_count_per_card = (
            df_join["card_id"].value_counts().mean().round(2).item()
        )
        retention_rate = round(
            df_join[df_join["state"] == 2]["rating"]
            .map({1: 0, 2: 1, 3: 1, 4: 1})
            .mean(),
            2,
        )

        result = {
            "user_id": user_id,
            "revlogs_count": revlogs_count,
            "card_count": card_count,
            "note_count": note_count,
            "avg_review_count_per_note": avg_review_count_per_note,
            "avg_review_count_per_card": avg_review_count_per_card,
            "retention_rate": retention_rate,
        }

        return result
    except Exception as e:
        print(f"Error processing user {user_id}: {str(e)}")
        return None


def process_users(user_ids, output_file="results.jsonl", max_workers=None):
    """
    Process multiple users in parallel and write results to JSONL file

    Args:
        user_ids: List of user IDs to process
        output_file: Path to output JSONL file
        max_workers: Maximum number of threads to use (None = auto)
    """
    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    # Process users in parallel with progress bar
    print(f"Processing {len(user_ids)} users...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(get_avg_review_count, user_ids),
                total=len(user_ids),
                desc="Processing users",
            )
        )

    # Filter out None results (failed processing)
    results = [r for r in results if r is not None]

    # Write results to JSONL file
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    # Example usage: process users 1 to 10
    user_ids = list(range(1, 10001))

    process_users(user_ids, output_file="results.jsonl", max_workers=10)

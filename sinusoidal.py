import pandas as pd
import numpy as np
import sys
import argparse

def generate_positional_embedding(position, embedding_dim=10):
    """
    Generate a positional embedding for a given position.
    
    Args:
        position (int): The position of the item in the series.
        embedding_dim (int): The dimensionality of the embedding vector.
        
    Returns:
        list: A list representing the positional embedding.
    """
    # Example positional embedding (sinusoidal encoding)
    # embedding = [np.sin(position / (10000 ** (2 * i / embedding_dim))) if i % 2 == 0 
    #              else np.cos(position / (10000 ** (2 * i / embedding_dim))) 
    #              for i in range(embedding_dim)]
    embedding = [
        float(np.sin(position / (10000 ** (2 * (i // 2) / embedding_dim)))) if i % 2 == 0 
        else float(np.cos(position / (10000 ** (2 * (i // 2) / embedding_dim))))
        for i in range(embedding_dim)
    ]
              
    return embedding
# Convert to DataFrame
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k', help='dataset name')
opt = parser.parse_args()
data = opt.data
df = pd.read_csv(f'Data/{data}.csv')
# Match new_data1.py's ID remapping so item_id aligns with the model's factorized ID space
df['item_id'] = pd.factorize(df['item_id'])[0]
#df=df.dropna()
# Helper function to extract series and position
def process_series(item_series_id):
    # Handle NaN / None / non-string values safely
    if not isinstance(item_series_id, str):
        return "Standalone", -1

    if item_series_id.startswith("SB"):  # Standalone item
        return "Standalone", -1
    elif item_series_id.startswith("Series_"):  # Series item, format: Series_<id>_P<position>
        parts = item_series_id.split("_")
        series_id = f"{parts[0]}_{parts[1]}"  # e.g. "Series_167817", "Series_MAL53", "Series_SS12431"
        position = int(parts[2][1:])  # "P3" -> 3
        return series_id, position
    return None, None

# Create the mapping table
def create_series_table(df):
    records = []
    for _, row in df.iterrows():
        series_id, position = process_series(row["item_series_id"])
        standalone_or_series = "Standalone" if series_id == "Standalone" else "Series"
        records.append({
            "item_id": row["item_id"],
            "item_series_id": row["item_series_id"],
            "series_id": series_id,
            "position_in_series": position,
            "standalone_or_series": standalone_or_series
        })
    return pd.DataFrame(records)

series_table = create_series_table(df)

# Add a new column 'positional_embedding' to the DataFrame
embedding_dim = 50  # Change this to your desired embedding dimension
series_table['positional_embedding'] = series_table.apply(
    lambda row: generate_positional_embedding(row['position_in_series'], embedding_dim) 
                if row['position_in_series'] != -1 else [0.0] * embedding_dim, 
    axis=1
)

series_table = series_table.drop_duplicates(subset=['item_id']).reset_index(drop=True)

series_table=series_table.drop('standalone_or_series',axis=1)
series_table = series_table[series_table['series_id'].str.startswith("Series", na=False)]

series_table.to_csv(f'Series_table_sinusodial_Series_table_{data}.csv', index=False)

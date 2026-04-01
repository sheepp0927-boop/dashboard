import pandas as pd
import os

# 1. Load the file you already trimmed (much faster!)
input_file = 'Crime_Dataset_Lite.csv' 
# NOTE: If you only have the zipped version locally, change this to 'Crime_Dataset_Lite.zip'

print(f"Loading {input_file}...")
df = pd.read_csv(input_file)

# 2. Convert Date to filter by Year
print("Processing dates...")
df['Date'] = pd.to_datetime(df['Date'])

# 3. FILTER: Keep only data from 2018 onwards
#    (Change to 2020 if you need it even smaller)
start_year = 2018
print(f"Keeping data from {start_year} to present...")
df_final = df[df['Date'].dt.year >= start_year].copy()

# 4. Print Stats
print(f"Rows before: {df.shape[0]}")
print(f"Rows after:  {df_final.shape[0]}")

# 5. Save as the final ZIP for GitHub
output_filename = 'Crime_Dataset_Lite.zip'
print(f"Saving to {output_filename}...")
df_final.to_csv(output_filename, index=False, compression='zip')

# 6. Check Size
size_mb = os.path.getsize(output_filename) / (1024 * 1024)
print(f"Done! Final File Size: {size_mb:.2f} MB")
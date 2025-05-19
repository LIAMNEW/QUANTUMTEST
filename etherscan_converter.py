import pandas as pd
import argparse
from datetime import datetime

def convert_etherscan_csv(input_file, output_file):
    """
    Converts Etherscan CSV export to the format required by our blockchain analyzer.
    
    Args:
        input_file: Path to the downloaded Etherscan CSV file
        output_file: Path where the converted CSV will be saved
    """
    print(f"Reading Etherscan data from {input_file}...")
    
    # Read the Etherscan CSV (column names may vary slightly)
    try:
        # Try with common Etherscan column names
        df = pd.read_csv(input_file)
        
        # Check if we have the necessary columns
        required_columns = ['Txhash', 'From', 'To', 'Value', 'DateTime']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Handle common Etherscan column name variations
        if missing_columns:
            if 'DateTime' in missing_columns and 'TimeStamp' in df.columns:
                df['DateTime'] = df['TimeStamp']
            if 'Value' in missing_columns and 'Value_IN(ETH)' in df.columns:
                df['Value'] = df['Value_IN(ETH)']
            elif 'Value' in missing_columns and 'Value_OUT(ETH)' in df.columns:
                df['Value'] = df['Value_OUT(ETH)']
            
            # Check again after fixes
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in input file: {missing_columns}")
                print(f"Available columns: {df.columns.tolist()}")
                return False
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return False
    
    # Create a new DataFrame with our required format
    new_df = pd.DataFrame()
    
    # Convert timestamp to our standard format
    print("Converting timestamps...")
    try:
        # Handle different timestamp formats from Etherscan
        if pd.api.types.is_numeric_dtype(df['DateTime']):
            # If timestamps are Unix timestamps (seconds since epoch)
            new_df['timestamp'] = pd.to_datetime(df['DateTime'], unit='s')
        else:
            # Try different datetime formats
            try:
                new_df['timestamp'] = pd.to_datetime(df['DateTime'])
            except:
                # Try specific format often used by Etherscan
                new_df['timestamp'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
        
        # Format as string in our standard format
        new_df['timestamp'] = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error converting timestamps: {str(e)}")
        # Use the original format as fallback
        new_df['timestamp'] = df['DateTime']
    
    # Copy address fields
    new_df['from_address'] = df['From']
    new_df['to_address'] = df['To']
    
    # Handle transaction value
    print("Processing transaction values...")
    try:
        # Convert to float and handle any text values
        new_df['value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # Fill any missing values with 0
        new_df['value'].fillna(0, inplace=True)
    except Exception as e:
        print(f"Error processing values: {str(e)}")
        new_df['value'] = 0
    
    # Add status column (Etherscan sometimes doesn't include this)
    if 'Status' in df.columns:
        new_df['status'] = df['Status']
    else:
        # Assume all transactions are confirmed
        new_df['status'] = 'confirmed'
    
    # Save the converted data
    print(f"Saving converted data to {output_file}...")
    new_df.to_csv(output_file, index=False)
    
    print(f"Successfully converted {len(new_df)} transactions!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Etherscan CSV to blockchain analyzer format')
    parser.add_argument('input_file', help='Path to the downloaded Etherscan CSV file')
    parser.add_argument('output_file', help='Path where the converted CSV will be saved')
    
    args = parser.parse_args()
    convert_etherscan_csv(args.input_file, args.output_file)
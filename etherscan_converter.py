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
        print(f"Available columns in the CSV: {df.columns.tolist()}")
        
        # Map columns based on available fields (handling common Etherscan format variations)
        column_map = {}
        
        # Handle timestamp/datetime variations
        if 'DateTime (UTC)' in df.columns:
            column_map['timestamp'] = 'DateTime (UTC)'
        elif 'DateTime' in df.columns:
            column_map['timestamp'] = 'DateTime'
        elif 'TimeStamp' in df.columns:
            column_map['timestamp'] = 'TimeStamp'
        elif 'UnixTimestamp' in df.columns:
            column_map['timestamp'] = 'UnixTimestamp'
            
        # Handle address fields
        if 'From' in df.columns:
            column_map['from_address'] = 'From'
        
        if 'To' in df.columns:
            column_map['to_address'] = 'To'
            
        # Handle value fields
        if 'Value' in df.columns:
            column_map['value'] = 'Value'
        elif 'Value_IN(ETH)' in df.columns and 'Value_OUT(ETH)' in df.columns:
            # Create a combined value column
            df['combined_value'] = df['Value_IN(ETH)'].astype(float) + df['Value_OUT(ETH)'].astype(float)
            column_map['value'] = 'combined_value'
        elif 'Value_IN(ETH)' in df.columns:
            column_map['value'] = 'Value_IN(ETH)'
        elif 'Value_OUT(ETH)' in df.columns:
            column_map['value'] = 'Value_OUT(ETH)'
            
        # Handle status field
        if 'Status' in df.columns:
            column_map['status'] = 'Status'
            
        # Check if we have the minimum required mappings
        required_mappings = ['timestamp', 'from_address', 'to_address']
        missing_mappings = [col for col in required_mappings if col not in column_map]
        
        if missing_mappings:
            print(f"Error: Could not map these required columns: {missing_mappings}")
            print("Please choose a different Etherscan export or try a different address.")
            return False
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return False
    
    # Create a new DataFrame with our required format
    new_df = pd.DataFrame()
    
    # Convert timestamp to our standard format
    print("Converting timestamps...")
    try:
        timestamp_col = column_map['timestamp']
        
        # Handle different timestamp formats from Etherscan
        if timestamp_col == 'UnixTimestamp':
            # If timestamps are Unix timestamps (seconds since epoch)
            new_df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='s')
        else:
            # Try different datetime formats
            try:
                new_df['timestamp'] = pd.to_datetime(df[timestamp_col])
            except:
                # Try specific format often used by Etherscan
                new_df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d %H:%M:%S')
        
        # Format as string in our standard format
        new_df['timestamp'] = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error converting timestamps: {str(e)}")
        # Create timestamps as fallback
        new_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Copy address fields
    new_df['from_address'] = df[column_map['from_address']]
    new_df['to_address'] = df[column_map['to_address']]
    
    # Handle transaction value
    print("Processing transaction values...")
    if 'value' in column_map:
        try:
            # Convert to float and handle any text values
            new_df['value'] = pd.to_numeric(df[column_map['value']], errors='coerce')
            
            # Fill any missing values with 0
            new_df['value'].fillna(0, inplace=True)
        except Exception as e:
            print(f"Error processing values: {str(e)}")
            new_df['value'] = 0
    else:
        # Use transaction fee as a proxy for value if available
        if 'TxnFee(ETH)' in df.columns:
            new_df['value'] = pd.to_numeric(df['TxnFee(ETH)'], errors='coerce')
        else:
            # No value available, use default values
            new_df['value'] = 0.1  # Small default value
    
    # Add status column
    if 'status' in column_map:
        new_df['status'] = df[column_map['status']]
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
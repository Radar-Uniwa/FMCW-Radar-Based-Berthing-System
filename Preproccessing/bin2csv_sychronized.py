"""
Input: folder with all the the radars measurements
Procedure: Parsing the data for all the radars seperatly. 16, 18 , 68 logic is from Texas and 14 is different
Output: Csv for each radar, with detected objects (x, y, z, azimut, elevation, dopppler, snr, noise)
"""

# import the required Python packages
import struct
import math
import binascii
import codecs
import csv
import os
import datetime
import re
import time
import subprocess
import pandas as pd
import argparse
import sys
import math


# Constants for parsing
TLV_HEADER_SIZE = 8  # TLV header size in bytes
OBJ_STRUCT_SIZE = 12  # Each object has 12 bytes of data
SYNC_KEY = b'\x02\x01\x04\x03\x06\x05\x08\x07'  # Known synchronization pattern for frames
SYNC_KEY_SIZE = len(SYNC_KEY)

def validate_sync_key_14(buffer, idx):
    return buffer[idx:idx + SYNC_KEY_SIZE] == SYNC_KEY

def find_next_sync_key_14(buffer, idx):
    while idx + SYNC_KEY_SIZE <= len(buffer):
        if validate_sync_key_14(buffer, idx):
            return idx
        idx += 1
    return len(buffer)

def parse_frame_header_1443(buffer, idx):
    header_format = '<QIIIIIII'
    header_size = struct.calcsize(header_format)
    if idx + header_size > len(buffer):
        return None, idx

    fields = struct.unpack_from(header_format, buffer, idx)
    header_info = {
        'sync_key': format(fields[0], '016x'),
        'version': fields[1],
        'length': fields[2],
        'platform': fields[3],
        'frame_number': fields[4],
        'cpu_cycles': fields[5],
        'num_objects': fields[6],
        'num_tlvs': fields[7]
    }
    return header_info, idx + header_size

def parse_tlv_data_1443(buffer, idx, configParameters):
    objects = []
    buffer_length = len(buffer)

    if not validate_sync_key_14(buffer, idx):
        idx = find_next_sync_key_14(buffer, idx)
        return objects, idx

    try:
        num_doppler_bins = configParameters["numDopplerBins"]
        doppler_resolution = configParameters["dopplerResolutionMps"]

        frame_header, idx = parse_frame_header_1443(buffer, idx)
        if frame_header is None:
            return objects, idx

        for _ in range(frame_header['num_tlvs']):
            if idx + TLV_HEADER_SIZE > buffer_length:
                break

            tlv_type, tlv_length = struct.unpack('<II', buffer[idx:idx + TLV_HEADER_SIZE])
            idx += TLV_HEADER_SIZE

            if tlv_type == 1:  # Detected objects TLV
                num_detected_obj, xyz_q_format = struct.unpack('<HH', buffer[idx:idx + 4])
                idx += 4
                for _ in range(num_detected_obj):
                    if idx + OBJ_STRUCT_SIZE > buffer_length:
                        break

                    range_i, doppler_i, peak_v, x, y, z = struct.unpack('<HHHhhh', buffer[idx:idx + OBJ_STRUCT_SIZE])
                    idx += OBJ_STRUCT_SIZE

                    mod_index = doppler_i % num_doppler_bins
                    if mod_index >= num_doppler_bins // 2:
                        doppler_i = mod_index - num_doppler_bins
                    else:
                        doppler_i = mod_index

                    doppler_val = doppler_i * doppler_resolution

                    x_coord = x / (1 << xyz_q_format)
                    y_coord = y / (1 << xyz_q_format)
                    z_coord = z / (1 << xyz_q_format)
                    
                    objects.append({
                        "x": x_coord,
                        "y": y_coord,
                        "z": z_coord,
                        "doppler": doppler_val,
                        "range": range_i,
                        "peak_value": peak_v
                    })

    except struct.error as e:
        print(f"Error parsing TLV data: {e}")

    return objects, idx

def find_bin_file_in_folder_14(folder_path):
    # Search for the first .bin file in the given folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".bin"):
            return os.path.join(folder_path, file_name)
    return None


# definations for parser pass/fail
TC_PASS   =  0
TC_FAIL   =  1

script_path = r"path/to/simple14parse_v2_incode_calling.py"
def getUint32(data):
    """!
       This function coverts 4 bytes to a 32-bit unsigned integer.

        @param data : 1-demension byte array  
        @return     : 32-bit unsigned integer
    """ 
    return (data[0] +
            data[1]*256 +
            data[2]*65536 +
            data[3]*16777216)

def getUint16(data):
    """!
       This function coverts 2 bytes to a 16-bit unsigned integer.

        @param data : 1-demension byte array
        @return     : 16-bit unsigned integer
    """ 
    return (data[0] +
            data[1]*256)

def getHex(data):
    """!
       This function coverts 4 bytes to a 32-bit unsigned integer in hex.

        @param data : 1-demension byte array
        @return     : 32-bit unsigned integer in hex
    """ 
    return (binascii.hexlify(data[::-1]))

def checkMagicPattern(data):
    """!
       This function check if data arrary contains the magic pattern which is the start of one mmw demo output packet.  

        @param data : 1-demension byte array
        @return     : 1 if magic pattern is found
                      0 if magic pattern is not found 
    """ 
    found = 0
    if (data[0] == 2 and data[1] == 1 and data[2] == 4 and data[3] == 3 and data[4] == 6 and data[5] == 5 and data[6] == 8 and data[7] == 7):
        found = 1
    return (found)

def parser_helper(data, readNumBytes):
    """!
       This function is called by parser_one_mmw_demo_output_packet() function or application to read the input buffer, find the magic number, header location, the length of frame, the number of detected object and the number of TLV contained in this mmw demo output packet.

        @param data                   : 1-demension byte array holds the the data read from mmw demo output. It ignorant of the fact that data is coming from UART directly or file read.  
        @param readNumBytes           : the number of bytes contained in this input byte array  
            
        @return headerStartIndex      : the mmw demo output packet header start location
        @return totalPacketNumBytes   : the mmw demo output packet lenght           
        @return numDetObj             : the number of detected objects contained in this mmw demo output packet          
        @return numTlv                : the number of TLV contained in this mmw demo output packet           
        @return subFrameNumber        : the sbuframe index (0,1,2 or 3) of the frame contained in this mmw demo output packet
    """ 

    headerStartIndex = -1

    for index in range (readNumBytes):
        if checkMagicPattern(data[index:index+8:1]) == 1:
            headerStartIndex = index
            break
        
    if headerStartIndex == -1: # does not find the magic number i.e output packet header 
        totalPacketNumBytes = -1
        numDetObj           = -1
        numTlv              = -1
        subFrameNumber      = -1
        platform            = -1
        frameNumber         = -1
        timeCpuCycles       = -1
    else: # find the magic number i.e output packet header 
        totalPacketNumBytes = getUint32(data[headerStartIndex+12:headerStartIndex+16:1])
        platform            = getHex(data[headerStartIndex+16:headerStartIndex+20:1])
        frameNumber         = getUint32(data[headerStartIndex+20:headerStartIndex+24:1])
        timeCpuCycles       = getUint32(data[headerStartIndex+24:headerStartIndex+28:1])
        numDetObj           = getUint32(data[headerStartIndex+28:headerStartIndex+32:1])
        numTlv              = getUint32(data[headerStartIndex+32:headerStartIndex+36:1])
        subFrameNumber      = getUint32(data[headerStartIndex+36:headerStartIndex+40:1])
        
    return (headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber)

def process_entire_bin_file(data, initial_posix_timestamp, frame_delay=0.05):
    """Process an entire binary file containing multiple radar frames."""
    readNumBytes = len(data)
    currentIndex = 0
    all_frames_results = []
    frame_idx = 0  # Frame index to calculate timestamp

    while currentIndex < readNumBytes:
        # Process one frame starting from the current index
        (
            result,
            headerStartIndex,
            totalPacketNumBytes,
            numDetObj,
            numTlv,
            subFrameNumber,
            detectedX_array,
            detectedY_array,
            detectedZ_array,
            detectedV_array,
            detectedRange_array,
            detectedAzimuth_array,
            detectedElevAngle_array,
            detectedSNR_array,
            detectedNoise_array,
        ) = parser_one_mmw_demo_output_packet(data[currentIndex:], readNumBytes - currentIndex)

        if result == TC_FAIL or headerStartIndex == -1:
            print("Failed to parse frame or no more valid frames.")
            break

        # Compute timestamp for the frame
        frame_timestamp = initial_posix_timestamp + frame_idx * frame_delay
        frame_idx += 1

        # Append the results of the current frame to the results list
        all_frames_results.append({
            "headerStartIndex": currentIndex + headerStartIndex,
            "totalPacketNumBytes": totalPacketNumBytes,
            "numDetObj": numDetObj,
            "numTlv": numTlv,
            "subFrameNumber": subFrameNumber,
            "detectedX_array": detectedX_array,
            "detectedY_array": detectedY_array,
            "detectedZ_array": detectedZ_array,
            "detectedV_array": detectedV_array,
            "detectedRange_array": detectedRange_array,
            "detectedAzimuth_array": detectedAzimuth_array,
            "detectedElevAngle_array": detectedElevAngle_array,
            "detectedSNR_array": detectedSNR_array,
            "detectedNoise_array": detectedNoise_array,
            "timestamp": frame_timestamp,
        })

        # Move the index to the next frame
        currentIndex += headerStartIndex + totalPacketNumBytes

        # If remaining data is too small to contain a full header, stop
        if readNumBytes - currentIndex < 40:
            break

    return all_frames_results



def parser_one_mmw_demo_output_packet(data, readNumBytes):
    """!
       This function is called by application. Firstly it calls parser_helper() function to find the start location of the mmw demo output packet, then extract the contents from the output packet.
       Each invocation of this function handles only one frame at a time and user needs to manage looping around to parse data for multiple frames.

        @param data                   : 1-demension byte array holds the the data read from mmw demo output. It ignorant of the fact that data is coming from UART directly or file read.  
        @param readNumBytes           : the number of bytes contained in this input byte array  
            
        @return result                : parser result. 0 pass otherwise fail
        @return headerStartIndex      : the mmw demo output packet header start location
        @return totalPacketNumBytes   : the mmw demo output packet lenght           
        @return numDetObj             : the number of detected objects contained in this mmw demo output packet          
        @return numTlv                : the number of TLV contained in this mmw demo output packet           
        @return subFrameNumber        : the sbuframe index (0,1,2 or 3) of the frame contained in this mmw demo output packet
        @return detectedX_array       : 1-demension array holds each detected target's x of the mmw demo output packet
        @return detectedY_array       : 1-demension array holds each detected target's y of the mmw demo output packet
        @return detectedZ_array       : 1-demension array holds each detected target's z of the mmw demo output packet
        @return detectedV_array       : 1-demension array holds each detected target's v of the mmw demo output packet
        @return detectedRange_array   : 1-demension array holds each detected target's range profile of the mmw demo output packet
        @return detectedAzimuth_array : 1-demension array holds each detected target's azimuth of the mmw demo output packet
        @return detectedElevAngle_array : 1-demension array holds each detected target's elevAngle of the mmw demo output packet
        @return detectedSNR_array     : 1-demension array holds each detected target's snr of the mmw demo output packet
        @return detectedNoise_array   : 1-demension array holds each detected target's noise of the mmw demo output packet
    """

    headerNumBytes = 40   

    PI = 3.14159265

    detectedX_array = []
    detectedY_array = []
    detectedZ_array = []
    detectedV_array = []
    detectedRange_array = []
    detectedAzimuth_array = []
    detectedElevAngle_array = []
    detectedSNR_array = []
    detectedNoise_array = []

    result = TC_PASS

    # call parser_helper() function to find the output packet header start location and packet size 
    (headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber) = parser_helper(data, readNumBytes)
                         
    if headerStartIndex == -1:
        result = TC_FAIL
    else:
        nextHeaderStartIndex = headerStartIndex + totalPacketNumBytes 

        if headerStartIndex + totalPacketNumBytes > readNumBytes:
            result = TC_FAIL
        elif nextHeaderStartIndex + 8 < readNumBytes and checkMagicPattern(data[nextHeaderStartIndex:nextHeaderStartIndex+8:1]) == 0:
            result = TC_FAIL 
        elif numDetObj < 0:
            result = TC_FAIL
        elif subFrameNumber > 3:
            result = TC_FAIL
        else: 
            # process the 1st TLV
            tlvStart = headerStartIndex + headerNumBytes
                                                    
            tlvType    = getUint32(data[tlvStart+0:tlvStart+4:1])
            tlvLen     = getUint32(data[tlvStart+4:tlvStart+8:1])       
            offset = 8
                    
                                                    
            # the 1st TLV must be type 1
            if tlvType == 1 and tlvLen < totalPacketNumBytes:#MMWDEMO_UART_MSG_DETECTED_POINTS
                         
                # TLV type 1 contains x, y, z, v values of all detect objects. 
                # each x, y, z, v are 32-bit float in IEEE 754 single-precision binary floating-point format, so every 16 bytes represent x, y, z, v values of one detect objects.    
                
                # for each detect objects, extract/convert float x, y, z, v values and calculate range profile and azimuth                           
                for obj in range(numDetObj):
                    # convert byte0 to byte3 to float x value
                    x = struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlvStart + offset:tlvStart + offset+4:1]),'hex'))[0]

                    # convert byte4 to byte7 to float y value
                    y = struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlvStart + offset+4:tlvStart + offset+8:1]),'hex'))[0]

                    # convert byte8 to byte11 to float z value
                    z = struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlvStart + offset+8:tlvStart + offset+12:1]),'hex'))[0]

                    # convert byte12 to byte15 to float v value
                    v = struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlvStart + offset+12:tlvStart + offset+16:1]),'hex'))[0]

                    # calculate range profile from x, y, z
                    compDetectedRange = math.sqrt((x * x)+(y * y)+(z * z))

                    # calculate azimuth from x, y           
                    if y == 0:
                        if x >= 0:
                            detectedAzimuth = 90
                        else:
                            detectedAzimuth = -90 
                    else:
                        detectedAzimuth = math.atan(x/y) * 180 / PI

                    # calculate elevation angle from x, y, z
                    if x == 0 and y == 0:
                        if z >= 0:
                            detectedElevAngle = 90
                        else: 
                            detectedElevAngle = -90
                    else:
                        detectedElevAngle = math.atan(z/math.sqrt((x * x)+(y * y))) * 180 / PI
                            
                    detectedX_array.append(x)
                    detectedY_array.append(y)
                    detectedZ_array.append(z)
                    detectedV_array.append(v)
                    detectedRange_array.append(compDetectedRange)
                    detectedAzimuth_array.append(detectedAzimuth)
                    detectedElevAngle_array.append(detectedElevAngle)
                                                                
                    offset = offset + 16
                # end of for obj in range(numDetObj) for 1st TLV
                                                            
            # Process the 2nd TLV
            tlvStart = tlvStart + 8 + tlvLen
                                                    
            tlvType    = getUint32(data[tlvStart+0:tlvStart+4:1])
            tlvLen     = getUint32(data[tlvStart+4:tlvStart+8:1])      
            offset = 8
                    
                                                            
            if tlvType == 7: 
                
                # TLV type 7 contains snr and noise of all detect objects.
                # each snr and noise are 16-bit integer represented by 2 bytes, so every 4 bytes represent snr and noise of one detect objects.    
            
                # for each detect objects, extract snr and noise                                            
                for obj in range(numDetObj):
                    # byte0 and byte1 represent snr. convert 2 bytes to 16-bit integer
                    snr   = getUint16(data[tlvStart + offset + 0:tlvStart + offset + 2:1])
                    # byte2 and byte3 represent noise. convert 2 bytes to 16-bit integer 
                    noise = getUint16(data[tlvStart + offset + 2:tlvStart + offset + 4:1])

                    detectedSNR_array.append(snr*0.1)
                    detectedNoise_array.append(noise*0.1)
                                                                    
                    offset = offset + 4
            else:
                for obj in range(numDetObj):
                    detectedSNR_array.append(0)
                    detectedNoise_array.append(0)

    return (result, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber, detectedX_array, detectedY_array, detectedZ_array, detectedV_array, detectedRange_array, detectedAzimuth_array, detectedElevAngle_array, detectedSNR_array, detectedNoise_array)

def read_bin_file(file_path):
    """Read a binary file and return its content as a byte array."""
    try:
        with open(file_path, 'rb') as bin_file:
            data = bin_file.read()
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    
def export_to_csv(frames_data, output_path):
    """Export parsed radar frames data to a CSV file."""
    with open(output_path, mode='w', newline='') as csv_file:
        # Define fieldnames
        fieldnames = ['Frame Number', 'POSIX Timestamp', 'Detected Objects']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx, frame in enumerate(frames_data, start=1):
            frame_timestamp = frame.get('timestamp', 0)
            detected_objects = []

            if 'objects' in frame:  # AWR1443 structure
                for obj in frame['objects']:
                    snr_value = 10 * math.log10(obj["peak_value"] + 1)  # Convert Peak Value to SNR
                    detected_objects.append(
                        f"[{obj['x']}, {obj['y']}, {obj['z']}, {obj['doppler']}, {snr_value}]"
                    )
            else:  # General radar structure
                for obj_idx in range(len(frame['detectedX_array'])):
                    detected_objects.append(
                        f"[{frame['detectedX_array'][obj_idx]}, {frame['detectedY_array'][obj_idx]}, "
                        f"{frame['detectedZ_array'][obj_idx]}, {frame['detectedV_array'][obj_idx]}, "
                        f"{frame['detectedSNR_array'][obj_idx]}]"
                    )

            # Join all objects with a comma, but without enclosing in another list
            detected_objects_str = ", ".join(detected_objects)

            # Write row to CSV
            writer.writerow({
                'Frame Number': frame_idx,
                'POSIX Timestamp': f"{frame_timestamp:.6f}",
                'Detected Objects': detected_objects_str,
            })

    print(f"Data exported to {output_path}")

def synchronize_csv_files(folder_path, expand_overlap_seconds=0.05):
    """
    Synchronize CSV files based on overlapping POSIX Timestamp ranges without creating a timestamp column.
    Each file is filtered to keep only the overlapping POSIX Timestamps.
    """
    # Identify all CSV files to synchronize
    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('_detected_objects.csv')
    ]

    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    # Read CSV files into DataFrames
    dataframes = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if not df.empty and 'POSIX Timestamp' in df.columns:
            dataframes[csv_file] = df
        else:
            print(f"Skipping file with invalid or missing 'POSIX Timestamp': {csv_file}")

    if not dataframes:
        print("No valid data to synchronize.")
        return None

    # Calculate overlapping POSIX Timestamp range
    latest_start = max(df['POSIX Timestamp'].min() for df in dataframes.values())
    earliest_stop = min(df['POSIX Timestamp'].max() for df in dataframes.values())

    print(f"Original overlapping range: {latest_start:.6f} to {earliest_stop:.6f}")

    # Expand the overlap slightly to account for timing mismatches
    latest_start -= expand_overlap_seconds
    earliest_stop += expand_overlap_seconds

    print(f"Expanded overlapping range: {latest_start:.6f} to {earliest_stop:.6f}")

    # Filter DataFrames based on the overlapping range
    output_folder = os.path.join(folder_path, 'synchronized_output')
    os.makedirs(output_folder, exist_ok=True)

    synchronized_files = {}
    for csv_file, df in dataframes.items():
        # Filter rows within the overlapping range
        filtered_df = df[(df['POSIX Timestamp'] >= latest_start) & (df['POSIX Timestamp'] <= earliest_stop)]
        output_file = os.path.join(output_folder, os.path.basename(csv_file))
        filtered_df.to_csv(output_file, index=False)
        synchronized_files[os.path.basename(csv_file)] = output_file
        print(f"Synchronized file saved to: {output_file}")

    print("Synchronization complete.")
    return synchronized_files



def print_timestamp_ranges(dataframes):
    """Print the timestamp range for each CSV file."""
    for key, df in dataframes.items():
        print(f"File: {key}")
        if 'timestamp' in df.columns:
            print(f"Min timestamp: {df['timestamp'].min()}, Max timestamp: {df['timestamp'].max()}")
            print(f"Rows before filtering: {len(df)}")
        else:
            print("No 'timestamp' column found.")


def merge_synchronized_csv_files_by_radar(folder_path):
    # Find all synchronized CSV files
    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('_detected_objects.csv')
    ]
    
    if len(csv_files) < 2:
        print("Not enough synchronized CSV files for merging.")
        return None

    # Mapping from identifier (extracted from the filename) to the desired radar header name
    radar_mapping = {
        "14": "AWR1443",
        "16": "AWR1642",
        "68": "AWR1843",
        "18": "IWR6843"
    }
    
    # Read each CSV and assign it a key based on the mapping
    dataframes = {}
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        parts = base_name.split('_')
        # Expecting the fourth element (index 3) to be the radar identifier, e.g., "14", "16", etc.
        if len(parts) > 3:
            radar_id = parts[3]
            radar_name = radar_mapping.get(radar_id, base_name)
        else:
            radar_name = base_name
        df = pd.read_csv(csv_file)
        if not df.empty and 'POSIX Timestamp' in df.columns:
            dataframes[radar_name] = df
        else:
            print(f"Skipping file with invalid or missing 'POSIX Timestamp': {csv_file}")

    # Use one of the dataframes as reference for timestamps
    reference_df = list(dataframes.values())[0]
    merged_data = []

    for frame_idx, row in reference_df.iterrows():
        posix_timestamp = row['POSIX Timestamp']
        frame_data = {'Frame Number': frame_idx + 1, 'POSIX Timestamp': posix_timestamp}
        
        # For each radar, get its detected objects within a 50ms tolerance
        for radar, df in dataframes.items():
            matching_rows = df[
                (df['POSIX Timestamp'] >= posix_timestamp - 0.05) &
                (df['POSIX Timestamp'] <= posix_timestamp + 0.05)
            ]
            if not matching_rows.empty:
                frame_data[radar] = matching_rows.iloc[0]['Detected Objects']
            else:
                frame_data[radar] = ''
                
        merged_data.append(frame_data)

    merged_df = pd.DataFrame(merged_data)
    output_file = os.path.join(folder_path, 'merged_detected_objects_by_radar.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV file saved to: {output_file}")
    return output_file


def main():
    try:
        # Prompt user for the measurement folder path
        measurement_folder = input("Please enter the full path to the measurement folder: ").strip().strip('"')
        if not os.path.exists(measurement_folder):
            print(f"Error: The folder path '{measurement_folder}' does not exist.")
            return

        # Define output folders
        csv_output_folder = os.path.join(measurement_folder, "csv_files")
        synchronized_output_folder = os.path.join(csv_output_folder, "synchronized_output")
        os.makedirs(csv_output_folder, exist_ok=True)
        os.makedirs(synchronized_output_folder, exist_ok=True)

        # Step 1: Process each subfolder to extract .bin files and create CSV files
        print("Processing subfolders to extract radar data...")
        subfolders = [os.path.join(measurement_folder, f) for f in os.listdir(measurement_folder) if os.path.isdir(os.path.join(measurement_folder, f))]
        frame_counts = {}
        
        for subfolder in subfolders:
            print(f"Processing subfolder: {subfolder}")

            # Search for .bin files in the subfolder
            bin_file = None
            for file_name in os.listdir(subfolder):
                if file_name.endswith(".bin"):
                    bin_file = os.path.join(subfolder, file_name)
                    break

            if not bin_file:
                print(f"No .bin file found in subfolder: {subfolder}")
                continue

            # Extract timestamp from the subfolder name
            timestamp_match = re.search(r'_(\d{8})_(\d{6})_(\d+)', subfolder)
            if timestamp_match:
                try:
                    date_str, time_str, nanoseconds_str = timestamp_match.groups()
                    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
                    hour, minute, second = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:])
                    nanoseconds = int(nanoseconds_str)

                    initial_timestamp = datetime.datetime(
                        year, month, day, hour, minute, second, nanoseconds // 1000
                    )
                    initial_posix_timestamp = time.mktime(initial_timestamp.timetuple()) + initial_timestamp.microsecond / 1e6
                except Exception as e:
                    print(f"Error parsing timestamp for subfolder '{subfolder}': {e}")
                    initial_posix_timestamp = 0
            else:
                print(f"No timestamp found in subfolder name: {subfolder}")
                initial_posix_timestamp = 0

            # Read the binary file
            binary_data = read_bin_file(bin_file)
            if not binary_data:
                print(f"Error reading binary file in subfolder: {subfolder}")
                continue

            # Detect platform and process the data
            platform = getHex(binary_data[0 + 16:0 + 20:1]) if 0 != -1 else "UNKNOWN"
            print(f"Detected platform: {platform}")

            if platform == b'000a1443':  # AWR1443 Radar
                print("Processing data for AWR1443 platform (skipping every second frame).")
                all_detected_objects = []
                idx = 0
                processed_frames_1443 = 0
                frame_count = 0  # Count all frames (processed and skipped)

                configParameters = {
                    "numDopplerBins": 8,
                    "dopplerResolutionMps": 0.13,
                    "rangeResolutionMeters": 0.244,
                    "maxRangeMeters": 50,
                    "maxRadialVelocityMps": 1
                }

                while idx < len(binary_data):
                    objects, idx = parse_tlv_data_1443(binary_data, idx, configParameters)

                    # Compute timestamp for the current frame
                    
                    # Process only every other frame
                    if frame_count % 2 == 0:
                        processed_frames_1443 += 1
                        frame_timestamp = initial_posix_timestamp + processed_frames_1443 * 0.05  # Increment by 50ms for every frame
                        all_detected_objects.append({'timestamp': frame_timestamp, 'objects': objects})

                    frame_count += 1  # Increment for every frame, not just processed frames

                # Save CSV for AWR1443
                frame_counts["AWR1443"] = len(all_detected_objects)
                csv_file_path = os.path.join(csv_output_folder, f"{os.path.splitext(os.path.basename(bin_file))[0]}_detected_objects.csv")
                export_to_csv(all_detected_objects, csv_file_path)

            else:  # General radar data
                print(f"Processing general radar data for platform: {platform}")
                all_frames_results = process_entire_bin_file(binary_data, initial_posix_timestamp)
                frame_counts[platform] = len(all_frames_results)

                # Define output CSV file name
                csv_file_name = os.path.splitext(os.path.basename(bin_file))[0] + "_detected_objects.csv"
                csv_file_path = os.path.join(csv_output_folder, csv_file_name)

                # Export data to CSV
                export_to_csv(all_frames_results, csv_file_path)

        # Step 2: Synchronize CSV files
        print("Synchronizing CSV files...")
        synchronized_data = synchronize_csv_files(csv_output_folder)

        if not synchronized_data:
            print("Synchronization failed or no synchronized data found.")
            return

        # Step 3: Merge synchronized CSV files
        print("Merging synchronized CSV files...")
        merged_csv_path = merge_synchronized_csv_files_by_radar(synchronized_output_folder)


        print("Processing completed successfully!")
        print(f"Frame counts for processed files: {frame_counts}")
        if merged_csv_path:
            print(f"Merged file saved at: {merged_csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    main()

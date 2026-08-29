import pandas as pd
import random
import os

def expand_tsv(input_file, output_file):
    # Read the original TSV file
    df = pd.read_csv(input_file, sep='\t')

    # Define clip parameters
    clip_duration = 5  # 5 seconds per clip
    stride = 2.5       # 2.5 seconds between clip starts (2.5s overlap)
    
    # List to hold all selected clips
    all_clips = []
    
    # Group by trial_type
    for trial_type, group in df.groupby('trial_type'):
        trial_clips = []
        # Generate clips for each segment in the group
        for index, row in group.iterrows():
            onset = row['onset']
            duration = row['duration']
            t = onset
            clip_idx = 0
            # Generate clips until the clip end exceeds segment end
            while t + clip_duration <= onset + duration:
                trial_clips.append({
                    'onset': t,
                    'duration': clip_duration,
                    'trial_type': trial_type + f"_{clip_idx:03d}"
                })
                t += stride
                clip_idx += 1
        # Sample 312 clips if possible, else take all
        if len(trial_clips) >= 312:
            sampled_clips = random.sample(trial_clips, 312)
        else:
            sampled_clips = trial_clips
        all_clips.extend(sampled_clips)
    
    # Create new DataFrame
    new_df = pd.DataFrame(all_clips)
    
    # Sort by onset (just in case)
    new_df = new_df.sort_values('onset')
    
    # Save to new TSV file
    new_df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
    print(f"Expanded TSV saved to {output_file}")
    print(f"Original rows: {len(df)}, New rows: {len(new_df)}")

if __name__ == "__main__":
    # Specify your input and output files
    subs = ['06', '08']
    run = ['01', '02']
    ses = ['01','02']

    for sub in subs:
        for r in run:
            for s in ses:
                input_file = f"/mnt/c/IDWB/ds_formal/sub-{sub}/ses-{s}/func/sub-{sub}_ses-{s}_task-roomtour_run-{r}_events.tsv"  # Change this to your input file name
                output_file = f"/mnt/c/IDWB/ds_formal/sub-{sub}/ses-{s}/func/sub-{sub}_ses-{s}_task-roomtour_run-{r}_events_aug.tsv"  # Change this to your desired output file name
                
                # Make sure input file exists
                if not os.path.exists(input_file):
                    print(f"Error: Input file '{input_file}' not found!")
                else:
                    expand_tsv(input_file, output_file)
    # input_file = "/mnt/c/IDWB/ds_formal/sub-02/ses-01/func/sub-02_ses-01_task-roomtour_run-01_events.tsv"  # Change this to your input file name
    # output_file = "/mnt/c/IDWB/ds_formal/sub-02/ses-01/func/sub-02_ses-01_task-roomtour_run-01_events_aug.tsv"  # Change this to your desired output file name
    
    # # Make sure input file exists
    # if not os.path.exists(input_file):
    #     print(f"Error: Input file '{input_file}' not found!")
    # else:
    #     expand_tsv(input_file, output_file)
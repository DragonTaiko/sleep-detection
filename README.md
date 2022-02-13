# sleep-detection

main.py is the file to detection
there are 2 functions

process_videos(datasource, video_path, results_path). This function analize videos in a path and creates files with the results.

sleep_detection(filesource, view_detection). This function analize one video, you pass file source and set if you want to view the detection per frame with True in the variable view_detection (with q you can skip the view detection). Also the function return 3 values, total frames in video, total frames with face detection and an array with states (0: awake, 1: sleep and -1: no face).

Analysis.ipybn is the file to get some metrics of results files.

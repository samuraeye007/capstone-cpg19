import numpy as np
import os
import cv2
import time

# Set paths
image_path = 'eng_heart'  # Path containing videos
output_path_deep = 'eng_heart/DeepFrames'
output_path_raw = 'eng_heart/RawFrames'
haarcascade_path = 'heartrate_model/haarcascade_frontalface_default.xml'

# Load the cascade
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Create output directories if they don't exist
os.makedirs(output_path_deep, exist_ok=True)
os.makedirs(output_path_raw, exist_ok=True)

# Get the list of videos
video_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
total_videos = len(video_files)
processed_count = 0

start_time = time.time()

for idx, video_name in enumerate(video_files, start=1):
    video_path = os.path.join(image_path, video_name)

    # Paths for processed frames
    deep_frames_path = os.path.join(output_path_deep, video_name)
    raw_frames_path = os.path.join(output_path_raw, video_name)

    # Skip video if both directories already exist
    if os.path.exists(deep_frames_path) and os.path.exists(raw_frames_path):
        print(f"Skipping already processed video: {video_name}")
        processed_count += 1
        continue

    # Create directories for processed frames
    os.makedirs(deep_frames_path, exist_ok=True)
    os.makedirs(raw_frames_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = nFrames

    L = 36
    C_R = np.empty((L, L, max_frames))
    C_G = np.empty((L, L, max_frames))
    C_B = np.empty((L, L, max_frames))

    D_R = np.empty((L, L, max_frames))
    D_G = np.empty((L, L, max_frames))
    D_B = np.empty((L, L, max_frames))

    D_R2 = np.empty((L, L, max_frames))
    D_G2 = np.empty((L, L, max_frames))
    D_B2 = np.empty((L, L, max_frames))

    medias_R = np.empty((L, L))
    medias_G = np.empty((L, L))
    medias_B = np.empty((L, L))

    desviaciones_R = np.empty((L, L))
    desviaciones_G = np.empty((L, L))
    desviaciones_B = np.empty((L, L))

    imagen = np.empty((L, L, 3))

    medias_CR = np.empty((L, L))
    medias_CG = np.empty((L, L))
    medias_CB = np.empty((L, L))

    desviaciones_CR = np.empty((L, L))
    desviaciones_CG = np.empty((L, L))
    desviaciones_CB = np.empty((L, L))

    ka = 1

    while cap.isOpened() and ka < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            break
        else:
            continue  # No face detected, skip this frame

        face = cv2.resize(face, (L, L), interpolation=cv2.INTER_AREA)
        C_R[:, :, ka] = face[:, :, 0]
        C_G[:, :, ka] = face[:, :, 1]
        C_B[:, :, ka] = face[:, :, 2]

        if ka > 1:
            D_R[:, :, ka - 1] = (C_R[:, :, ka] - C_R[:, :, ka - 1]) / (C_R[:, :, ka] + C_R[:, :, ka - 1])
            D_G[:, :, ka - 1] = (C_G[:, :, ka] - C_G[:, :, ka - 1]) / (C_G[:, :, ka] + C_G[:, :, ka - 1])
            D_B[:, :, ka - 1] = (C_B[:, :, ka] - C_B[:, :, ka - 1]) / (C_B[:, :, ka] + C_B[:, :, ka - 1])
        ka += 1

    for i in range(L):
        for j in range(L):
            medias_R[i, j] = np.mean(D_R[i, j, :])
            medias_G[i, j] = np.mean(D_G[i, j, :])
            medias_B[i, j] = np.mean(D_B[i, j, :])
            desviaciones_R[i, j] = np.std(D_R[i, j, :])
            desviaciones_G[i, j] = np.std(D_G[i, j, :])
            desviaciones_B[i, j] = np.std(D_B[i, j, :])

    for i in range(L):
        for j in range(L):
            medias_CR[i, j] = np.mean(C_R[i, j, :])
            medias_CG[i, j] = np.mean(C_G[i, j, :])
            medias_CB[i, j] = np.mean(C_B[i, j, :])
            desviaciones_CR[i, j] = np.std(C_R[i, j, :])
            desviaciones_CG[i, j] = np.std(C_G[i, j, :])
            desviaciones_CB[i, j] = np.std(C_B[i, j, :])

    # Save RawFrames
    for k in range(max_frames):
        D_R2[:, :, k] = (C_R[:, :, k] - medias_CR) / (desviaciones_CR + 0.1)
        D_G2[:, :, k] = (C_G[:, :, k] - medias_CG) / (desviaciones_CG + 0.1)
        D_B2[:, :, k] = (C_B[:, :, k] - medias_CB) / (desviaciones_CB + 0.1)

        imagen[:, :, 0] = D_R2[:, :, k]
        imagen[:, :, 1] = D_G2[:, :, k]
        imagen[:, :, 2] = D_B2[:, :, k]

        imagen = np.clip(imagen, 0, 255).astype(np.uint8)

        nombre_salvar = os.path.join(raw_frames_path, f"{k}.png")
        cv2.imwrite(nombre_salvar, imagen)

    # Save DeepFrames
    for k in range(max_frames):
        D_R[:, :, k] = (D_R[:, :, k] - medias_R) / (desviaciones_R + 0.1)
        D_G[:, :, k] = (D_G[:, :, k] - medias_G) / (desviaciones_G + 0.1)
        D_B[:, :, k] = (D_B[:, :, k] - medias_B) / (desviaciones_B + 0.1)

        imagen[:, :, 0] = D_R[:, :, k]
        imagen[:, :, 1] = D_G[:, :, k]
        imagen[:, :, 2] = D_B[:, :, k]

        imagen = np.clip(imagen, 0, 255).astype(np.uint8)

        nombre_salvar = os.path.join(deep_frames_path, f"{k}.png")
        cv2.imwrite(nombre_salvar, imagen)

    cap.release()
    cv2.destroyAllWindows()

    # Update processed count and print progress
    processed_count += 1
    elapsed_time = time.time() - start_time
    avg_time_per_video = elapsed_time / processed_count
    remaining_videos = total_videos - processed_count
    estimated_time_remaining = avg_time_per_video * remaining_videos

    print(f"Processed {processed_count}/{total_videos} videos.")
    print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds.")

print("Processing complete.")

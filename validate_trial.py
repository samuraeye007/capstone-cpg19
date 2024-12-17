import argparse
import torch
import numpy as np
from data import AVLip
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
from collections import defaultdict
import time

def validate(model, loader, gpu_id, frame_output="results/frame_scores.txt", avg_output="results/lipmotion_scores.txt"):
    print("Validating...")
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    total_batches = len(loader)
    start_time = time.time()  # Start timing

    with torch.no_grad():
        video_scores = defaultdict(list)  # Store scores grouped by video path
        y_true, y_pred, img_paths = [], [], []

        # Open frame output file
        with open(frame_output, "w") as frame_file:
            for idx, (img, crops, label) in enumerate(loader):
                # Time estimation
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / (idx + 1)
                remaining_time = avg_time_per_batch * (total_batches - (idx + 1))

                print(
                    f"Processing batch {idx + 1}/{total_batches} "
                    f"- Elapsed time: {elapsed_time:.2f}s, "
                    f"Estimated time remaining: {remaining_time:.2f}s"
                )

                # Get batch paths
                batch_paths = loader.dataset.total_list[
                    idx * loader.batch_size : min((idx + 1) * loader.batch_size, len(loader.dataset))
                ]

                # Move data to the appropriate device
                img_tens = img.to(device)
                crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
                features = model.get_features(img_tens).to(device)

                # Model prediction
                pred = model(crops_tens, features)[0].sigmoid().flatten().tolist()
                y_pred.extend(pred)
                y_true.extend(label.flatten().tolist())
                img_paths.extend(batch_paths)

                # Write frame-by-frame scores
                for path, score in zip(batch_paths, pred):
                    frame_file.write(f"{path}: {score:.4f}\n")
                    frame_file.flush()

                # Group scores by video (use full path for grouping)
                for path, score in zip(batch_paths, pred):
                    video_name = "/".join(path.split("/")[:-1])  # Use the directory path
                    video_scores[video_name].append(score)

        # Compute average scores for each video
        video_avg_scores = {video: sum(scores) / len(scores) for video, scores in video_scores.items()}

        # Write average scores to file
        with open(avg_output, "w") as avg_file:
            for video, avg_score in video_avg_scores.items():
                avg_file.write(f"{avg_score:.4f}\n")
            print(f"Average video scores saved to {avg_output}")

    # Additional Metrics (optional)
    y_true = np.array(y_true)
    y_pred_bin = np.where(np.array(y_pred) >= 0.5, 1, 0)

    # Calculate metrics
    ap = average_precision_score(y_true, y_pred_bin)
    cm = confusion_matrix(y_true, y_pred_bin)
    tp, fn, fp, tn = cm.ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = accuracy_score(y_true, y_pred_bin)
    
    return ap, fpr, fnr, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_list_path", type=str, default="eng_processed/0_real")
    parser.add_argument("--fake_list_path", type=str, default="eng_processed/1_fake")
    parser.add_argument("--max_sample", type=int, default=1000, help="Max number of validate samples")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--data_label", type=str, default="val")
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/ckpt.pth")
    parser.add_argument("--gpu", type=int, default=0)

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA {opt.gpu} for inference.")

    model = build_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    print("Model loaded.")
    model.eval()
    model.to(device)

    dataset = AVLip(opt)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False
    )
    ap, fpr, fnr, acc = validate(
        model, loader, gpu_id=[opt.gpu], frame_output="results/frame_scores.txt", avg_output="results/lipmotion_scores.txt"
    )
    print(f"Accuracy: {acc:.4f}, Average Precision: {ap:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")

import cv2
import torch
import os

def detect(source='0',
           weights='runs/train/military_yolo/weights/best.pt',
           conf_thres=0.25,
           project_dir='runs/train/military_yolo'):
    # Determine weight file path and fallback to last.pt if best.pt is missing
    best_path = weights
    last_path = os.path.join(project_dir, 'weights', 'last.pt')
    if not os.path.exists(best_path):
        if os.path.exists(last_path):
            print(f"Warning: {best_path} not found, using last.pt instead.")
            best_path = last_path
        else:
            raise FileNotFoundError(
                f"Neither best.pt nor last.pt found in {project_dir}/weights. "
                "Please train the model first.")

    # Load model with custom weights
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom', path=best_path, force_reload=True
    )
    model.conf = conf_thres  # Set confidence threshold

    # Initialize video capture
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)

    save_path = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)

        # Render results
        annotated = results.render()[0]

        # Save annotated image for file sources (once)
        if save_path is None and not source.isdigit():
            filename = os.path.basename(source)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(os.getcwd(), f"{name}_annotated{ext}")
            cv2.imwrite(save_path, annotated)

        cv2.imshow('Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='camera index or file path')
    parser.add_argument(
        '--weights', type=str,
        default='runs/train/military_yolo/weights/best.pt',
        help='path to model weights'
    )
    parser.add_argument(
        '--conf', type=float, default=0.25, help='confidence threshold'
    )
    parser.add_argument(
        '--project', type=str,
        default='runs/train/military_yolo',
        help='project directory where weights are stored'
    )
    args = parser.parse_args()
    output = detect(
        source=args.source,
        weights=args.weights,
        conf_thres=args.conf,
        project_dir=args.project
    )
    if output:
        print(f"Annotated file saved to: {output}")
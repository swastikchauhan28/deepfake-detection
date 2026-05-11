import cv2, os, numpy as np

def make_video(path, frames=40, fake=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (224,224))
    for i in range(frames):
        frame = np.random.randint(0, 255, (224,224,3), dtype=np.uint8)
        if fake:
            cv2.putText(frame, 'FAKE', (50,112),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        out.write(frame)
    out.release()

for split in ['train', 'val', 'test']:
    for i in range(30):
        make_video(f'data/{split}/real/real_{i:03d}.mp4', fake=False)
        make_video(f'data/{split}/fake/fake_{i:03d}.mp4', fake=True)
    print(f'{split}: 30 real + 30 fake videos created')

print('\nDone! Dataset ready for training.')
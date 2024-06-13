# 여기서 모델, 인풋, 아웃풋 경로 지정 (ppe, hpe)
# !python run_infer.py 만 실행시켜서 inference.py 돌리기
# run_infer.py 하나로 ppe랑 hpe가 한번씩 돌려서 각자 저장되게

import subprocess

def run_inference(model_file, files_loc, output_dir):
    command = [
        'python', 'inference.py',
        '--model-file', model_file,
        '--files-loc', files_loc,
        '--output-dir', output_dir
    ]
    subprocess.run(command, check=True)

def main():
    # hpe
    hpe_model_file = '/content/drive/MyDrive/OmniPose/checkpointhpe.pth'
    hpe_files_loc = '/content/drive/MyDrive/infer_hpe'
    hpe_output_dir = '/content/drive/MyDrive/samples/infer_hpe_out'

    # ppe
    ppe_model_file = '/content/drive/MyDrive/OmniPose/checkpointppe.pth'
    ppe_files_loc = '/content/drive/MyDrive/infer_ppe'
    ppe_output_dir = '/content/drive/MyDrive/samples/infer_ppe_out'

    # hpe inference 실행
    run_inference(hpe_model_file, hpe_files_loc, hpe_output_dir)

    # ppe inference 실행
    run_inference(ppe_model_file, ppe_files_loc, ppe_output_dir)

if __name__ == '__main__':
    main()
# 여기서 모델, 인풋, 아웃풋 경로 지정 (ppe, hpe)
# !python run_infer.py 만 실행시켜서 inference.py 돌리기
# run_infer.py 하나로 ppe랑 hpe가 한번씩 돌려서 각자 저장되게

import subprocess

def run_inference(model_file, files_loc, output_dir, generation_id):
    command = [
        'python', 'inference.py', 
        '--model-file', model_file, 
        '--files-loc', files_loc, 
        '--output-dir', output_dir,
        '--generation-id', generation_id  # generation_id를 인자로 추가
    ]
    subprocess.run(command, check=True)

def run_infer(generationId, conditionImageUrl, targetImageUrl):
    ppe_model_file = '/content/drive/MyDrive/OmniPose/checkpoint_ppe.pth'  # 경로 수정
    ppe_files_loc = targetImageUrl
    ppe_output_dir = f'/content/drive/MyDrive/samples/ppe/{generationId}'

    hpe_model_file = '/content/drive/MyDrive/OmniPose/checkpoint_hpe.pth'  # 경로 수정
    hpe_files_loc = conditionImageUrl
    hpe_output_dir = f'/content/drive/MyDrive/samples/hpe/{generationId}'

    run_inference(ppe_model_file, ppe_files_loc, ppe_output_dir, generationId)
    run_inference(hpe_model_file, hpe_files_loc, hpe_output_dir, generationId)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on multiple models')
    parser.add_argument('--generation-id', required=True, help='Generation ID for the inference run')
    parser.add_argument('--target-image-url', required=True, help='Path to the PPE input image directory')
    parser.add_argument('--condition-image-url', required=True, help='Path to the HPE input image directory')

    args = parser.parse_args()

    run_infer(args.generation_id, args.condition_image_url, args.target_image_url)

import os 
import json
import argparse
from pathlib import Path

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/sl/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/sl/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    results = evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
    os.makedirs("results", exist_ok=True)
    landlord_cfg = Path(args.landlord).parent.name
    peasant_cfg = Path(args.landlord_up).parent.name
    output_path = f"results/{landlord_cfg}__vs__{peasant_cfg}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation JSON to {output_path}")

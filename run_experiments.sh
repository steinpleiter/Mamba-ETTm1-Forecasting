cd "$(dirname "$0")"

# Run the complete pipeline
python scripts/run_all.py --device mps --epochs 30 --batch_size 32

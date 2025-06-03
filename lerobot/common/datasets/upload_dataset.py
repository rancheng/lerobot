#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.utils import validate_repo_id

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import create_lerobot_dataset_card

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Upload a LeRobot dataset to the Hugging Face Hub')
    parser.add_argument('--repo-id', type=str, required=True,
                       help='The repository ID on the Hugging Face Hub (e.g., "username/dataset-name")')
    parser.add_argument('--root', type=str, default=None,
                       help='Local directory containing the dataset. If not specified, uses default cache directory.')
    parser.add_argument('--branch', type=str, default=None,
                       help='Branch name to upload to. If not specified, uses default branch.')
    parser.add_argument('--tags', type=str, nargs='+', default=None,
                       help='List of tags to add to the dataset')
    parser.add_argument('--license', type=str, default='apache-2.0',
                       help='License for the dataset (default: apache-2.0)')
    parser.add_argument('--private', action='store_true',
                       help='Whether to make the repository private')
    parser.add_argument('--no-push-videos', action='store_true',
                       help='Skip uploading video files')
    parser.add_argument('--large-folder', action='store_true',
                       help='Use upload_large_folder for large datasets')
    parser.add_argument('--description', type=str, default=None,
                       help='Description to add to the dataset card')
    parser.add_argument('--paper-url', type=str, default=None,
                       help='URL to the paper associated with this dataset')
    parser.add_argument('--homepage', type=str, default=None,
                       help='Homepage URL for the dataset')
    
    return parser.parse_args()

def validate_args(args):
    try:
        validate_repo_id(args.repo_id)
    except ValueError as e:
        logger.error(f"Invalid repository ID: {e}")
        raise

    if args.root:
        root_path = Path(args.root)
        if not root_path.exists():
            raise ValueError(f"Dataset root directory does not exist: {args.root}")

def main():
    args = parse_args()
    validate_args(args)

    logger.info(f"Loading dataset from {args.root or 'default cache directory'}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root
    )

    # Prepare card kwargs
    card_kwargs = {
        'description': args.description,
        'paper_url': args.paper_url,
        'homepage': args.homepage
    }
    # Remove None values
    card_kwargs = {k: v for k, v in card_kwargs.items() if v is not None}

    logger.info(f"Pushing dataset to the Hub: {args.repo_id}")
    try:
        dataset.push_to_hub(
            branch=args.branch,
            tags=args.tags,
            license=args.license,
            tag_version=True,
            push_videos=not args.no_push_videos,
            private=args.private,
            upload_large_folder=args.large_folder,
            **card_kwargs
        )
        logger.info("Dataset upload completed successfully!")
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise

if __name__ == '__main__':
    main() 
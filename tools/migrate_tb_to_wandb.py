#!/usr/bin/env python3
"""
TensorBoard to Wandb Migration Script for DeepfakeBench

Migrates historical training logs from TensorBoard format to Wandb,
preserving experiment organization, timestamps, and metric hierarchies.

Author: Claude Agent  
Date: 2026-03-24
"""

import os
import sys
import yaml
import pickle
import logging
import argparse
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import wandb
from tqdm import tqdm

# Add parent dirs to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("WARNING: TensorBoard not available. Install with: pip install tensorboard")


class TensorBoardToWandbMigrator:
    """Main migration class for converting TensorBoard logs to Wandb"""

    def __init__(self, config_path: str):
        """Initialize migrator with configuration"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.migration_state = self._load_migration_state()

        # Validate TensorBoard availability
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is required for migration. Install with: pip install tensorboard>=2.20.0")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for migration process"""
        logger = logging.getLogger('wandb_migration')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _load_config(self, config_path: str) -> Dict:
        """Load migration configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Set defaults
            migration_config = config.get('migration', {})
            defaults = {
                'source_dir': './logs/training/',
                'wandb': {
                    'project': 'deepfakebench-historical',
                    'entity': None,
                    'tags': ['historical', 'migrated']
                },
                'experiments': 'all',
                'include_predictions': False,
                'include_checkpoints': False,
                'batch_size': 100,
                'max_retries': 3,
                'rate_limit_delay': 0.1,
                'progress_tracking': True,
                'log_file': './logs/migration.log'
            }

            # Merge with defaults
            for key, value in defaults.items():
                if key not in migration_config:
                    migration_config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in migration_config[key]:
                            migration_config[key][subkey] = subvalue

            return migration_config

        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def _load_migration_state(self) -> Dict:
        """Load or initialize migration state for resumability"""
        state_file = Path(self.config.get('log_file', './logs/migration.log')).with_suffix('.state.json')

        if state_file.exists():
            try:
                import json
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load migration state: {e}")

        return {
            'completed_experiments': [],
            'failed_experiments': [],
            'current_progress': {},
            'last_checkpoint': None
        }

    def _save_migration_state(self):
        """Save current migration state"""
        state_file = Path(self.config.get('log_file', './logs/migration.log')).with_suffix('.state.json')
        state_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            import json
            with open(state_file, 'w') as f:
                json.dump(self.migration_state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save migration state: {e}")

    def parse_experiment_timestamp(self, experiment_dir: str) -> Optional[datetime]:
        """Extract timestamp from experiment directory name"""
        # Pattern: model_YYYY-MM-DD-HH-MM-SS
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$', experiment_dir)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')
            except ValueError as e:
                self.logger.warning(f"Could not parse timestamp from {experiment_dir}: {e}")
        return None

    def load_experiment_config(self, log_dir: Path) -> Dict:
        """Parse training.log for experiment configuration"""
        training_log = log_dir / 'training.log'
        config = {}

        if not training_log.exists():
            self.logger.warning(f"No training.log found in {log_dir}")
            return config

        try:
            with open(training_log, 'r') as f:
                content = f.read()

            # Extract key configuration parameters
            # Model name
            model_match = re.search(r'model_name[\'\"]\s*:\s*[\'\"](.*?)[\'\"]', content)
            if model_match:
                config['model_name'] = model_match.group(1)

            # Datasets
            dataset_match = re.search(r'train_dataset[\'\"]\s*:\s*\[(.*?)\]', content)
            if dataset_match:
                datasets = [d.strip().strip('\'"') for d in dataset_match.group(1).split(',')]
                config['train_dataset'] = datasets

            # Learning rate
            lr_match = re.search(r'lr[\'\"]\s*:\s*([0-9e\-\.]+)', content)
            if lr_match:
                config['learning_rate'] = float(lr_match.group(1))

            # Batch size
            batch_match = re.search(r'batch_size[\'\"]\s*:\s*(\d+)', content)
            if batch_match:
                config['batch_size'] = int(batch_match.group(1))

        except Exception as e:
            self.logger.error(f"Failed to parse training.log in {log_dir}: {e}")

        return config

    def parse_tb_events(self, event_file_path: Path) -> Dict[str, List[Tuple[int, float]]]:
        """Parse TensorBoard event files and extract scalar metrics"""
        if not TENSORBOARD_AVAILABLE:
            self.logger.error("TensorBoard not available for event parsing")
            return {}

        try:
            ea = EventAccumulator(str(event_file_path))
            ea.Reload()

            scalars = {}
            scalar_tags = ea.Tags().get('scalars', [])

            for tag in scalar_tags:
                try:
                    scalar_events = ea.Scalars(tag)
                    scalars[tag] = [(event.step, event.value) for event in scalar_events]
                except Exception as e:
                    self.logger.warning(f"Failed to extract scalar '{tag}' from {event_file_path}: {e}")

            return scalars

        except Exception as e:
            self.logger.error(f"Failed to parse TensorBoard events in {event_file_path}: {e}")
            return {}

    def load_pickle_metrics(self, pickle_path: Path) -> Dict:
        """Load metrics from pickle files"""
        if not pickle_path.exists():
            return {}

        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)

            # Extract key metrics, optionally skip prediction arrays
            metrics = {}

            # Always include scalar metrics
            for key in ['acc', 'auc', 'eer', 'ap', 'video_auc']:
                if key in data:
                    metrics[key] = data[key]

            # Optionally include prediction arrays
            if self.config.get('include_predictions', False):
                for key in ['pred', 'label']:
                    if key in data and isinstance(data[key], np.ndarray):
                        # Convert to list for JSON serialization
                        metrics[key] = data[key].tolist()

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to load pickle file {pickle_path}: {e}")
            return {}

    def find_experiment_files(self, experiment_dir: Path) -> Dict[str, List[Path]]:
        """Find all relevant files in an experiment directory"""
        files = {
            'tensorboard_events': [],
            'pickle_files': [],
            'checkpoints': [],
            'training_log': None
        }

        # Find training.log
        training_log = experiment_dir / 'training.log'
        if training_log.exists():
            files['training_log'] = training_log

        # Find TensorBoard event files
        for event_file in experiment_dir.rglob('events.out.tfevents.*'):
            files['tensorboard_events'].append(event_file)

        # Find pickle files
        for pickle_file in experiment_dir.rglob('*.pickle'):
            files['pickle_files'].append(pickle_file)

        # Find checkpoints
        if self.config.get('include_checkpoints', False):
            for ckpt_file in experiment_dir.rglob('*.pth'):
                files['checkpoints'].append(ckpt_file)

        return files

    def create_historical_run(self, experiment_name: str, timestamp: datetime,
                            config_data: Dict, dry_run: bool = False) -> Optional[Any]:
        """Create a wandb run for historical experiment"""
        if dry_run:
            self.logger.info(f"[DRY RUN] Would create wandb run: {experiment_name}")
            return None

        try:
            # Extract model type for tagging
            model_type = experiment_name.split('_')[0] if '_' in experiment_name else experiment_name

            # Create tags
            tags = self.config['wandb']['tags'].copy()
            tags.append(model_type)

            run = wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                dir="./logs/wandb/",
                name=f"{experiment_name}_historical",
                tags=tags,
                config=config_data,
                resume='never'
            )

            # Set historical metadata
            run.summary.update({
                'original_timestamp': timestamp.isoformat(),
                'migration_date': datetime.now().isoformat(),
                'data_source': 'tensorboard_migration',
                'experiment_name': experiment_name
            })

            return run

        except Exception as e:
            self.logger.error(f"Failed to create wandb run for {experiment_name}: {e}")
            return None

    def process_experiment_metrics(self, files: Dict, experiment_name: str) -> Dict:
        """Process all metrics from an experiment"""
        all_metrics = defaultdict(list)

        # Process TensorBoard events
        for event_file in files['tensorboard_events']:
            self.logger.debug(f"Processing event file: {event_file}")

            # Determine metric context from path
            # Path structure: experiment/train|test/dataset/metric/metric_board/events.out.tfevents.*
            path_parts = event_file.parts

            phase = None
            dataset = None
            metric_type = None

            # Find phase (train/test)
            for i, part in enumerate(path_parts):
                if part in ['train', 'test']:
                    phase = part
                    if i + 1 < len(path_parts):
                        dataset = path_parts[i + 1]
                    if i + 2 < len(path_parts):
                        metric_type = path_parts[i + 2]
                    break

            if not phase:
                self.logger.warning(f"Could not determine phase from path: {event_file}")
                continue

            # Parse events
            scalars = self.parse_tb_events(event_file)

            # Convert to wandb format
            for tag, events in scalars.items():
                for step, value in events:
                    # Create wandb metric name
                    if phase == 'train':
                        wandb_key = f"train/{tag}"
                    else:
                        wandb_key = f"test/{dataset}/{tag}"

                    all_metrics[wandb_key].append((step, value))

        # Process pickle files for final metrics
        final_metrics = {}
        for pickle_file in files['pickle_files']:
            if 'metric_dict_best.pickle' in pickle_file.name:
                # Determine dataset from path
                path_parts = pickle_file.parts
                dataset = None
                for i, part in enumerate(path_parts):
                    if part == 'test' and i + 1 < len(path_parts):
                        dataset = path_parts[i + 1]
                        break

                metrics = self.load_pickle_metrics(pickle_file)
                if dataset and metrics:
                    for metric_name, value in metrics.items():
                        if not isinstance(value, (list, np.ndarray)):  # Skip arrays unless configured
                            wandb_key = f"test/{dataset}/final/{metric_name}"
                            final_metrics[wandb_key] = value

        return {'time_series': dict(all_metrics), 'final': final_metrics}

    def migrate_experiment(self, experiment_dir: Path, dry_run: bool = False) -> bool:
        """Migrate a single experiment to wandb"""
        experiment_name = experiment_dir.name

        # Skip if already completed (unless doing dry run)
        if not dry_run and experiment_name in self.migration_state['completed_experiments']:
            self.logger.info(f"Skipping already migrated experiment: {experiment_name}")
            return True

        mode_str = "[DRY RUN] " if dry_run else ""
        self.logger.info(f"{mode_str}Migrating experiment: {experiment_name}")

        try:
            # Parse timestamp
            timestamp = self.parse_experiment_timestamp(experiment_name)
            if not timestamp:
                timestamp = datetime.now()  # Fallback

            # Load experiment configuration
            config_data = self.load_experiment_config(experiment_dir)

            # Find all relevant files
            files = self.find_experiment_files(experiment_dir)

            self.logger.info(f"Found {len(files['tensorboard_events'])} event files, "
                           f"{len(files['pickle_files'])} pickle files, "
                           f"{len(files['checkpoints'])} checkpoints")

            # Create wandb run
            run = self.create_historical_run(experiment_name, timestamp, config_data, dry_run)

            if not dry_run and run:
                # Process and upload metrics
                metrics = self.process_experiment_metrics(files, experiment_name)

                # Reorganize time series data by step to ensure monotonic logging
                time_series_data = metrics['time_series']
                step_to_metrics = defaultdict(dict)

                # Group all metrics by their step number
                for metric_name, events in time_series_data.items():
                    for step, value in events:
                        step_to_metrics[step][metric_name] = value

                # Log all metrics at each step together, in step order
                for step in sorted(step_to_metrics.keys()):
                    wandb.log(step_to_metrics[step], step=step)

                    # Rate limiting
                    if self.config['rate_limit_delay'] > 0:
                        import time
                        time.sleep(self.config['rate_limit_delay'])

                # Upload final metrics
                if metrics['final']:
                    wandb.log(metrics['final'])

                # Upload checkpoints if configured
                if self.config.get('include_checkpoints', False) and files['checkpoints']:
                    for ckpt_file in files['checkpoints']:
                        try:
                            wandb.save(str(ckpt_file))
                        except Exception as e:
                            self.logger.warning(f"Failed to upload checkpoint {ckpt_file}: {e}")

                # Finish run
                wandb.finish()

            # Mark as completed (only if not dry run)
            if not dry_run:
                self.migration_state['completed_experiments'].append(experiment_name)
                self._save_migration_state()

            self.logger.info(f"Successfully migrated experiment: {experiment_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to migrate experiment {experiment_name}: {e}")
            # Only track failures if not dry run
            if not dry_run:
                if experiment_name not in self.migration_state['failed_experiments']:
                    self.migration_state['failed_experiments'].append(experiment_name)
                self._save_migration_state()
            return False

    def get_experiment_list(self) -> List[Path]:
        """Get list of experiments to migrate"""
        source_dir = Path(self.config['source_dir'])

        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Find experiment directories
        experiment_dirs = []
        for item in source_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it looks like an experiment directory
                if any(item.glob('training.log')) or any(item.glob('**/*.tfevents.*')):
                    experiment_dirs.append(item)

        # Filter experiments if specified
        experiments_config = self.config.get('experiments', 'all')
        if experiments_config != 'all' and isinstance(experiments_config, list):
            filtered_dirs = []
            for exp_dir in experiment_dirs:
                if exp_dir.name in experiments_config:
                    filtered_dirs.append(exp_dir)
            experiment_dirs = filtered_dirs

        # Sort by timestamp for consistent processing
        experiment_dirs.sort(key=lambda x: x.name)

        return experiment_dirs

    def batch_migrate_all(self, dry_run: bool = False) -> Dict[str, int]:
        """Migrate all experiments"""
        experiments = self.get_experiment_list()

        if not experiments:
            self.logger.warning("No experiments found to migrate")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}

        mode_str = "[DRY RUN] " if dry_run else ""
        self.logger.info(f"{mode_str}Found {len(experiments)} experiments to migrate")

        results = {'total': len(experiments), 'success': 0, 'failed': 0, 'skipped': 0}

        # Process experiments with progress bar
        desc = "[DRY RUN] Migrating experiments" if dry_run else "Migrating experiments"
        progress_bar = tqdm(experiments, desc=desc) if self.config.get('progress_tracking', True) else experiments

        for experiment_dir in progress_bar:
            # Skip already completed experiments (but not during dry run)
            if not dry_run and experiment_dir.name in self.migration_state['completed_experiments']:
                results['skipped'] += 1
                continue

            success = self.migrate_experiment(experiment_dir, dry_run)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1

        # Log summary
        self.logger.info(f"Migration completed: {results['success']} success, "
                        f"{results['failed']} failed, {results['skipped']} skipped")

        return results


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Migrate TensorBoard logs to Wandb')
    parser.add_argument('--config', '-c', required=True,
                       help='Path to migration configuration YAML file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without uploading to wandb')
    parser.add_argument('--experiment', '-e',
                       help='Migrate specific experiment only')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger('wandb_migration').setLevel(logging.DEBUG)

    try:
        migrator = TensorBoardToWandbMigrator(args.config)

        if args.experiment:
            # Migrate specific experiment
            experiment_path = Path(migrator.config['source_dir']) / args.experiment
            if not experiment_path.exists():
                print(f"Experiment not found: {experiment_path}")
                return 1

            success = migrator.migrate_experiment(experiment_path, args.dry_run)
            return 0 if success else 1
        else:
            # Migrate all experiments
            results = migrator.batch_migrate_all(args.dry_run)

            # Print summary
            print("\n" + "="*50)
            print("MIGRATION SUMMARY")
            print("="*50)
            print(f"Total experiments: {results['total']}")
            print(f"Successfully migrated: {results['success']}")
            print(f"Failed: {results['failed']}")
            print(f"Skipped (already done): {results['skipped']}")

            return 0 if results['failed'] == 0 else 1

    except Exception as e:
        print(f"Migration failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

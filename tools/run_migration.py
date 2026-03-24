#!/usr/bin/env python3
"""
Simple CLI for running TensorBoard to Wandb migration

Usage examples:
  python run_migration.py                           # Migrate all experiments
  python run_migration.py --dry-run                 # Test run without uploading
  python run_migration.py --experiment effort_2026-03-20-00-25-30  # Single experiment
  python run_migration.py --test-one                # Test with smallest experiment
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from migrate_tb_to_wandb import TensorBoardToWandbMigrator
except ImportError as e:
    print(f"Failed to import migration script: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run TensorBoard to Wandb migration',
        epilog='Examples:\n'
               '  %(prog)s                           # Migrate all experiments\n'
               '  %(prog)s --dry-run                 # Test without uploading\n'
               '  %(prog)s --experiment EXPERIMENT   # Migrate specific experiment\n'
               '  %(prog)s --test-one                # Test with smallest experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        default='migration_config.yaml',
        help='Migration configuration file (default: migration_config.yaml)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Perform dry run without uploading to wandb'
    )
    parser.add_argument(
        '--experiment', '-e',
        help='Migrate specific experiment only (e.g., effort_2026-03-20-00-25-30)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Skip predictions and checkpoints for faster migration'
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Make sure migration_config.yaml exists in the tools/ directory")
        return 1

    # Show configuration
    print("🔧 Migration Configuration:")
    print(f"   Config: {config_path}")
    if args.experiment:
        print(f"   Single experiment: {args.experiment}")
    else:
        print("   Mode: Batch migration (all experiments)")
    print(f"   Dry run: {'Yes' if args.dry_run else '🚨 No - will upload to wandb!'}")
    print(f"   Verbose: {'Yes' if args.verbose else 'No'}")

    # Confirm if not dry run
    if not args.dry_run:
        response = input("\n⚠️  This will upload historical data to wandb. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cancelled")
            return 0

    print("\n" + "="*50)
    print("🚀 Starting Migration")
    print("="*50)

    try:
        # Run migration
        from migrate_tb_to_wandb import main as migration_main
        
        # Build args for migration script
        migration_args = ['--config', str(config_path)]
        if args.dry_run:
            migration_args.append('--dry-run')
        if args.experiment:
            migration_args.extend(['--experiment', args.experiment])
        if args.verbose:
            migration_args.append('--verbose')

        # Temporarily replace sys.argv
        original_argv = sys.argv[:]
        sys.argv = ['migrate_tb_to_wandb.py'] + migration_args
        
        result = migration_main()
        
        # Restore sys.argv
        sys.argv = original_argv

        if result == 0:
            print("\n✅ Migration completed successfully!")
            if args.dry_run:
                print("💡 Run without --dry-run to actually upload to wandb")
        else:
            print("\n❌ Migration failed")

        return result

    except Exception as e:
        print(f"\n❌ Migration error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""Add RSM root to sys.path so all imports resolve without install."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

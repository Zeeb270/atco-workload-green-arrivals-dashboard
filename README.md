# ATCO Workload-Aware Green Arrival Dashboard

This project develops a machine-learning and LLM-enabled dashboard for analysing workload-aware sustainable airport arrival operations. The prototype extracts operational complexity features from arrival traffic data, predicts air traffic controller workload risk, compares baseline and green arrival strategies, and uses an LLM assistant to explain operational trade-offs.

## Motivation

Airport arrival procedures must balance efficiency, environmental performance, safety, and controller workload. A strategy that reduces holding or fuel burn may still be operationally unsuitable if it increases traffic complexity or air traffic controller workload.

This project explores how machine learning can support decision-making by predicting workload risk from arrival traffic complexity features and visualizing the trade-off between delay, environmental proxy metrics, and workload.

## Research Context

The project is inspired by recent research on predicting air traffic controller workload using eye-tracking and machine learning. In that work, workload was estimated using physiological and behavioral indicators such as pupil diameter, blink behavior, fixations, saccades, and head movement.

This repository adapts the idea to an aviation-logistics setting: instead of eye-tracking features, it uses operational traffic-complexity features extracted from arrival scenarios.

## Main Features

- Upload or load arrival traffic data
- Extract traffic-complexity features in three-minute time windows
- Predict low, medium, or high workload using machine-learning models
- Compare baseline, green, and workload-aware arrival strategies
- Visualize workload over time, delay, emissions proxy, and feature importance
- Use an LLM assistant to explain dashboard results in natural language

## System Architecture

```text
Arrival traffic data
        ↓
Feature engineering in time windows
        ↓
Machine-learning workload prediction
        ↓
Strategy comparison
        ↓
Interactive dashboard
        ↓
LLM explanation layer

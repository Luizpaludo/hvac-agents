"""
ranker.py - Recommendation library and the learning ranker (RL element, owner: Luiz Paludo).

Design decision: the LLM NEVER invents maintenance actions. Candidate actions
come from this reviewed library (wording based on the Trane / NIST / ASHRAE
manuals in the RAG library). The agent learns WHICH candidate to put first.

Reinforcement learning (from the proposal, section 3):
  reward r in [-1, +1]: +1 accepted / confirmed, +0.5 modified, -0.5 rejected or
  false alarm, -1 missed fault or comfort/safety violation.
  Update per category c (fault type) and candidate a:
      Q(c,a) <- Q(c,a) + 0.2 * (r - Q(c,a))
  Ranking: sort by Q (exploit) but with 10 % probability use a random order
  (explore), an epsilon-greedy multi-armed bandit policy.
Q-values live in the q_values table, so learning survives between runs.
"""
import random

from .. import config

# parameter/value are what the Safety Reviewer checks against config.LIMITS.
# evidence_terms: words a manual passage must contain to support this fault.
LIBRARY = {
    "stuck_oa_damper": {
        "evidence_terms": ["damper"],
        "candidates": [
            {"id": "dmp-1", "text": "Inspect the outdoor air damper actuator and linkage for a stuck or disconnected blade; compare damper command with actual position."},
            {"id": "dmp-2", "text": "Check the economizer control sequence: the outdoor air damper should return to minimum position when outdoor air is warmer than return air."},
            {"id": "dmp-3", "text": "Lower the supply air temperature setpoint to 12 C until the damper is repaired.",
             "parameter": "supply_air_temp_setpoint", "value": 12.0},
        ]},
    "slipping_fan_belt": {
        "evidence_terms": ["belt"],
        "candidates": [
            {"id": "blt-1", "text": "Inspect supply fan belt tension and wear; re-tension or replace belts as described in the air handler maintenance manual."},
            {"id": "blt-2", "text": "Check sheave alignment and set-screw torque; misaligned sheaves cause belt slip."},
            {"id": "blt-3", "text": "Raise the duct static pressure setpoint to 3.0 inWC to compensate for low airflow.",
             "parameter": "duct_static_pressure_setpoint", "value": 3.0},
        ]},
    "ccv_stuck_closed": {
        "evidence_terms": ["valve"],
        "candidates": [
            {"id": "ccv-1", "text": "Inspect the cooling coil valve and actuator for a stuck-closed or leaking valve; verify the valve strokes with the controller output."},
            {"id": "ccv-2", "text": "Check chilled water supply temperature and flow to the coil (pump status, chilled water availability)."},
            {"id": "ccv-3", "text": "Check the controller output range configuration for the cooling coil valve (e.g. 0-10 V vs 2-10 V)."},
        ]},
    "mat_sensor_bias": {
        "evidence_terms": ["sensor", "mixed air"],
        "candidates": [
            {"id": "mat-1", "text": "Verify the mixed air temperature sensor against a calibrated handheld thermometer; recalibrate or replace if biased."},
            {"id": "mat-2", "text": "Check mixed air sensor location (averaging element across the duct) for stratification."},
            {"id": "mat-3", "text": "Trend outdoor, return and mixed air temperatures for 24 h to confirm the sensor fault before replacing parts."},
        ]},
    "excess_energy_use": {
        "evidence_terms": ["energy", "schedul", "economizer"],
        "candidates": [
            {"id": "nrg-1", "text": "Review the HVAC operating schedule: confirm air handlers and fans shut down or set back after occupied hours."},
            {"id": "nrg-2", "text": "Check for simultaneous heating and cooling or economizer faults during the flagged period."},
            {"id": "nrg-3", "text": "Raise the night zone temperature setpoint to 28 C to save energy.",
             "parameter": "zone_temp", "value": 28.0},
        ]},
    "ccv_stuck": {
        "evidence_terms": ["valve"],
        "candidates": [
            {"id": "cst-1", "text": "Inspect the cooling coil valve actuator and linkage: the valve position does not follow the controller command (stuck valve)."},
            {"id": "cst-2", "text": "Stroke the cooling coil valve manually from 0 to 100 % and compare the feedback signal with the command."},
            {"id": "cst-3", "text": "Check chilled water supply temperature and flow to the coil while the valve is being repaired."},
        ]},
    "ccv_leakage": {
        "evidence_terms": ["valve", "leak"],
        "candidates": [
            {"id": "lk-1", "text": "Check the cooling coil valve seat for leakage: the valve does not close fully when the controller asks for 0 %."},
            {"id": "lk-2", "text": "Check the valve actuator zero/stroke calibration so that a 0 % command fully closes the valve."},
            {"id": "lk-3", "text": "Compare coil leaving air temperature with entering air temperature when the valve is commanded closed to confirm the leak."},
        ]},
    "sat_sensor_bias": {
        "evidence_terms": ["sensor", "supply air"],
        "candidates": [
            {"id": "sab-1", "text": "Verify the supply air temperature sensor against a calibrated handheld thermometer; recalibrate or replace if biased."},
            {"id": "sab-2", "text": "Check the supply air sensor location and wiring (e.g. too close to the coil or a loose connection)."},
            {"id": "sab-3", "text": "Trend cooling valve command against mixed and supply air temperatures for 24 h to confirm the sensor fault."},
        ]},
    "oat_sensor_bias": {
        "evidence_terms": ["sensor", "outdoor air", "mixed air"],
        "candidates": [
            {"id": "oab-1", "text": "Verify the outdoor air temperature sensor against a reference (weather station or handheld) and recalibrate if biased."},
            {"id": "oab-2", "text": "Check that the outdoor air sensor is shaded and away from exhaust air or direct sun."},
            {"id": "oab-3", "text": "Review economizer decisions during the flagged period: a biased outdoor sensor makes the economizer open or close at the wrong time."},
        ]},
    "unknown": {
        "evidence_terms": ["fault"],
        "candidates": [
            {"id": "unk-1", "text": "Review the trend data of the flagged window with a qualified engineer; no single fault rule matched."},
            {"id": "unk-2", "text": "Check recent sensor calibration and controller changes for the affected air handler."},
            {"id": "unk-3", "text": "Schedule a walk-down inspection of the air handler during the next maintenance window."},
        ]},
}

FAULT_QUERIES = {
    "stuck_oa_damper": "outdoor air damper stuck economizer mixed air temperature fault",
    "slipping_fan_belt": "slipping supply fan belt tension low static pressure",
    "ccv_stuck_closed": "cooling coil valve stuck supply air temperature above setpoint",
    "mat_sensor_bias": "mixed air temperature sensor bias outside outdoor and return air range",
    "excess_energy_use": "energy waste schedule simultaneous heating cooling economizer",
    "ccv_stuck": "cooling coil valve stuck valve position does not follow command",
    "ccv_leakage": "cooling coil valve leaking does not close fully",
    "sat_sensor_bias": "supply air temperature sensor bias offset calibration",
    "oat_sensor_bias": "outdoor air temperature sensor bias economizer",
    "unknown": "air handling unit fault detection and diagnosis",
}


class Ranker:
    def __init__(self, memory, rng=None, epsilon=config.EPSILON, lr=config.LEARNING_RATE):
        self.memory, self.epsilon, self.lr = memory, epsilon, lr
        self.rng = rng or random.Random(config.SEED)

    def rank(self, category, explore=True):
        """Return the candidates of a category, best first, each with its Q-value."""
        cands = [dict(c) for c in LIBRARY.get(category, LIBRARY["unknown"])["candidates"]]
        q = self.memory.get_q(category)
        for c in cands:
            c["q_value"] = round(q.get(c["id"], (0.0, 0))[0], 3)
        if explore and self.rng.random() < self.epsilon:
            self.rng.shuffle(cands)
            for c in cands:
                c["explored"] = True
            return cands
        return sorted(cands, key=lambda c: -c["q_value"])   # stable: library order breaks ties

    def update(self, category, candidate_id, reward):
        q, n = self.memory.get_q(category).get(candidate_id, (0.0, 0))
        q_new = q + self.lr * (reward - q)
        self.memory.set_q(category, candidate_id, q_new, n + 1)
        return q_new

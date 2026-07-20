# ============================================================
# generate_reports.py
# Generates realistic simulated Daily Drilling Reports (DDRs)
# for testing our Q&A system.
#
# Run inside Docker:
# docker exec -it drilling-report-qa python generate_reports.py
# ============================================================

import os           # For file and folder operations
import json         # For saving metadata as JSON
from datetime import datetime, timedelta  # For date handling

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

# Output folder where reports will be saved
OUTPUT_DIR = "data/raw_reports"

# Well information (fictional but realistic)
WELL_INFO = {
    "well_name": "Well Alpha-7",
    "field": "North Desert Field",
    "operator": "PetroCo Exploration Ltd.",
    "rig_name": "Rig Desert Star 3",
    "spud_date": "2024-01-15",
    "target_depth": 12500,   # feet
    "location": "Block 7, Sector North, Desert Basin"
}

# -------------------------------------------------------
# REPORT DATA
# Each entry represents one day of drilling operations
# All values are realistic for a typical onshore well
# -------------------------------------------------------

DAILY_DATA = [
    # DAY 1
    {
        "day": 1,
        "date": "2024-01-15",
        "depth_start": 0,
        "depth_end": 312,
        "bit_size": "26 inch",
        "bit_type": "Tricone",
        "wob": 15,           # Weight on Bit in klbs
        "rpm": 80,           # Rotations per minute
        "rop": 78,           # Rate of Penetration in ft/hr
        "mud_weight": 8.6,   # ppg (pounds per gallon)
        "mud_viscosity": 42, # Funnel viscosity in seconds
        "pump_pressure": 650, # psi
        "ecd": 8.9,          # Equivalent Circulating Density ppg
        "formation": "Surface alluvial deposits and unconsolidated sands",
        "incidents": None,
        "npt_hours": 0,
        "operations": [
            "06:00 - Rigged up and commenced spud operations",
            "08:30 - Drilled 26-inch surface hole with water",
            "14:00 - Encountered first sand formation at 180 feet",
            "18:00 - Switched to mud system, mud weight 8.6 ppg",
            "24:00 - Total depth reached 312 feet, all operations normal"
        ],
        "safety_notes": "Pre-tour safety meeting conducted. All personnel wore appropriate PPE.",
        "mud_report": "Fresh water mud system. Viscosity 42 sec. pH 9.2. Fluid loss 8 cc/30min."
    },

    # DAY 2
    {
        "day": 2,
        "date": "2024-01-16",
        "depth_start": 312,
        "depth_end": 895,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 22,
        "rpm": 120,
        "rop": 58,
        "mud_weight": 9.2,
        "mud_viscosity": 48,
        "pump_pressure": 1850,
        "ecd": 9.6,
        "formation": "Shale interbedded with thin sandstone stringers",
        "incidents": None,
        "npt_hours": 0,
        "operations": [
            "06:00 - Ran and cemented 20-inch surface casing to 310 feet",
            "10:00 - WOC (Waiting on Cement) for 4 hours",
            "14:00 - Rigged up 17.5-inch PDC bit and commenced drilling",
            "16:00 - Entered shale formation at 380 feet",
            "24:00 - Drilled to 895 feet, good ROP maintained throughout"
        ],
        "safety_notes": "Cement job completed successfully. BOP tested to 3000 psi — passed.",
        "mud_report": "KCl polymer mud. Weight 9.2 ppg. Viscosity 48 sec. pH 9.5."
    },

    # DAY 3
    {
        "day": 3,
        "date": "2024-01-17",
        "depth_start": 895,
        "depth_end": 1456,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 24,
        "rpm": 115,
        "rop": 52,
        "mud_weight": 9.4,
        "mud_viscosity": 51,
        "pump_pressure": 1920,
        "ecd": 9.8,
        "formation": "Grey shale with occasional limestone intercalations",
        "incidents": "Tight hole at 1,200 feet — reamed for 45 minutes",
        "npt_hours": 0.75,
        "operations": [
            "06:00 - Continued drilling 17.5-inch hole section",
            "09:30 - Encountered tight hole at 1,200 feet",
            "09:30 - Picked up off bottom, reamed tight section for 45 minutes",
            "10:15 - Resumed normal drilling operations",
            "14:00 - Encountered limestone intercalation at 1,310 feet, ROP dropped to 28 ft/hr",
            "18:00 - Drilled through limestone, ROP recovered to 55 ft/hr in shale below",
            "24:00 - Depth 1,456 feet. All parameters normal."
        ],
        "safety_notes": "H2S monitors checked and calibrated. Readings zero throughout shift.",
        "mud_report": "KCl polymer mud. Weight 9.4 ppg. Increased due to formation pressure gradient."
    },

    # DAY 4
    {
        "day": 4,
        "date": "2024-01-18",
        "depth_start": 1456,
        "depth_end": 1821,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 25,
        "rpm": 110,
        "rop": 45,
        "mud_weight": 9.6,
        "mud_viscosity": 54,
        "pump_pressure": 2100,
        "ecd": 10.1,
        "formation": "Dense limestone formation",
        "incidents": "STUCK PIPE — Differential sticking at 1,654 feet",
        "npt_hours": 3.5,
        "operations": [
            "06:00 - Continued drilling through dense limestone section",
            "08:15 - ROP dropped significantly to 18 ft/hr in hard limestone",
            "11:30 - Pipe became differentially stuck at 1,654 feet depth",
            "11:30 - Shut down pumps, attempted to free pipe by rotation — unsuccessful",
            "12:00 - Applied 50,000 lbs overpull. No movement detected.",
            "12:30 - Spotted 30 barrels of diesel spotting fluid around stuck point",
            "13:45 - Resumed slow rotation at 40 RPM with diesel soak",
            "15:00 - Pipe freed successfully after 3.5 hours NPT",
            "15:00 - Inspected pipe, no damage found",
            "15:30 - Resumed normal drilling operations",
            "24:00 - Drilled to 1,821 feet. Mud weight increased to 9.6 ppg preventatively."
        ],
        "safety_notes": "Stuck pipe emergency procedure followed correctly. All crew accounted for during incident.",
        "mud_report": "Mud weight increased to 9.6 ppg. Added diesel spotting fluid — 30 bbls used."
    },

    # DAY 5
    {
        "day": 5,
        "date": "2024-01-19",
        "depth_start": 1821,
        "depth_end": 2387,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 23,
        "rpm": 118,
        "rop": 54,
        "mud_weight": 9.6,
        "mud_viscosity": 52,
        "pump_pressure": 2050,
        "ecd": 10.0,
        "formation": "Interbedded sandstone and shale — possible reservoir indicators",
        "incidents": None,
        "npt_hours": 0,
        "operations": [
            "06:00 - Continued drilling. Formation changed to interbedded sands and shales.",
            "08:00 - Gas detector reading increased from 0 to 45 units at 1,950 feet",
            "08:00 - Informed company man and drilling supervisor of gas show",
            "08:30 - Increased mud weight from 9.6 to 9.8 ppg as precaution",
            "09:00 - Gas readings stabilized at 12 units, drilling continued",
            "14:00 - Encountered oil staining in sandstone cuttings at 2,200 feet",
            "14:00 - Notified geologist. Samples collected and bagged for analysis.",
            "18:00 - Strong oil show at 2,280 feet. Cuttings fluorescence confirmed.",
            "24:00 - Drilled to 2,387 feet. Excellent day with potential reservoir encountered."
        ],
        "safety_notes": "Gas show protocol activated at 08:00. All non-essential personnel moved to muster point briefly. Situation controlled.",
        "mud_report": "Mud weight increased to 9.8 ppg due to gas show. Chlorides stable at 4,200 ppm."
    },

    # DAY 6
    {
        "day": 6,
        "date": "2024-01-20",
        "depth_start": 2387,
        "depth_end": 2391,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 0,
        "rpm": 0,
        "rop": 0,
        "mud_weight": 10.2,
        "mud_viscosity": 58,
        "pump_pressure": 0,
        "ecd": 0,
        "formation": "Top of reservoir sand — logging while drilling",
        "incidents": "Lost circulation at 2,389 feet — 45 barrels lost",
        "npt_hours": 14.0,
        "operations": [
            "06:00 - Ran wireline logs over potential reservoir section 2,200-2,387 feet",
            "06:00 - Logging operations: GR, resistivity, neutron, density logs run",
            "10:00 - Log results show 42 feet of net pay in sandstone reservoir",
            "10:00 - Excellent porosity 22-28%, resistivity values indicate oil saturation",
            "12:00 - Resumed drilling below logged section",
            "12:15 - Sudden loss of returns at 2,389 feet",
            "12:15 - Lost 45 barrels of mud to formation in 15 minutes",
            "12:30 - Pumped 30 sacks of LCM (Lost Circulation Material) pill",
            "14:00 - Partial returns restored, losses reduced to 5 bbls/hr",
            "18:00 - Pumped second LCM pill — 50 sacks bentonite/diesel mixture",
            "22:00 - Losses stopped completely, full returns established",
            "24:00 - Total depth remains 2,391 feet. 14 hours NPT due to lost circulation."
        ],
        "safety_notes": "Lost circulation contingency plan activated. No safety incidents during LCM operations.",
        "mud_report": "Mud weight increased to 10.2 ppg. Lost 45 bbls total. LCM pills successful."
    },

    # DAY 7
    {
        "day": 7,
        "date": "2024-01-21",
        "depth_start": 2391,
        "depth_end": 3102,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 26,
        "rpm": 112,
        "rop": 48,
        "mud_weight": 10.2,
        "mud_viscosity": 56,
        "pump_pressure": 2180,
        "ecd": 10.6,
        "formation": "Tight carbonate with anhydrite streaks",
        "incidents": None,
        "npt_hours": 0,
        "operations": [
            "06:00 - Resumed normal drilling after lost circulation resolved",
            "08:00 - Bit changed at surface — previous bit showed 70% wear (dull grade: 2-3-WT)",
            "10:00 - New PDC bit (616 type) run in hole",
            "12:00 - Resumed drilling at 2,391 feet",
            "14:00 - Entered tight carbonate section, ROP reduced to 32 ft/hr",
            "18:00 - Anhydrite streaks encountered at 2,650 feet, increased torque noted",
            "22:00 - Drilled through anhydrite, formation returned to carbonate",
            "24:00 - Total depth 3,102 feet. Good progress despite tight formation."
        ],
        "safety_notes": "Bit trip completed safely. Rotary table guard in place during all operations.",
        "mud_report": "Mud weight held at 10.2 ppg. Viscosity 56 sec. Sulfate content monitored due to anhydrite."
    },

    # DAY 8
    {
        "day": 8,
        "date": "2024-01-22",
        "depth_start": 3102,
        "depth_end": 3587,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 28,
        "rpm": 105,
        "rop": 41,
        "mud_weight": 10.4,
        "mud_viscosity": 59,
        "pump_pressure": 2340,
        "ecd": 10.9,
        "formation": "Deep shale with increasing pressure gradient",
        "incidents": "Pump failure on Mud Pump #2 — 2 hours downtime",
        "npt_hours": 2.0,
        "operations": [
            "06:00 - Continued drilling deep shale section",
            "07:00 - Mud Pump #2 failed due to worn liner — shut down immediately",
            "07:00 - Switched to single pump operation (Mud Pump #1) at reduced flow rate",
            "07:15 - Maintenance crew began repairing Pump #2",
            "09:00 - Pump #2 repaired and returned to service — new liner installed",
            "09:00 - Resumed dual pump operation, normal flow rate restored",
            "10:00 - Pressure gradient increasing, mud weight raised to 10.4 ppg",
            "14:00 - Torque fluctuations noted — possible formation instability",
            "16:00 - Added 10 ppb PHPA polymer to mud system to stabilize shale",
            "24:00 - Depth 3,587 feet. Formation pressure being carefully monitored."
        ],
        "safety_notes": "Pump repair conducted safely. Pressure relief valves checked before resuming operations.",
        "mud_report": "Mud weight 10.4 ppg. Added PHPA polymer for shale stabilization. Funnel viscosity 59 sec."
    },

    # DAY 9
    {
        "day": 9,
        "date": "2024-01-23",
        "depth_start": 3587,
        "depth_end": 4012,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 27,
        "rpm": 108,
        "rop": 38,
        "mud_weight": 10.6,
        "mud_viscosity": 62,
        "pump_pressure": 2420,
        "ecd": 11.1,
        "formation": "Transition zone — hard limestone capping potential deep reservoir",
        "incidents": None,
        "npt_hours": 1.0,
        "operations": [
            "06:00 - Drilling continued in deep shale transitioning to limestone",
            "08:00 - MWD (Measurement While Drilling) tool malfunction — data gap 1 hour",
            "08:00 - Reduced drilling parameters during MWD data gap for safety",
            "09:00 - MWD tool recovered and restarted, data stream restored",
            "12:00 - Top of deep limestone formation encountered at 3,780 feet",
            "12:00 - ROP dropped from 45 to 22 ft/hr in hard limestone",
            "14:00 - Increased WOB to 30 klbs to maintain ROP in hard formation",
            "18:00 - Gas readings increased to 85 units — possible deeper reservoir",
            "18:00 - Mud weight increased to 10.6 ppg, informed company man",
            "22:00 - Gas readings stabilized at 25 units after mud weight increase",
            "24:00 - Depth 4,012 feet. Approaching potential deep reservoir target."
        ],
        "safety_notes": "Well control awareness briefing given to all crew at 18:00 due to gas increase. BOP ready.",
        "mud_report": "Mud weight 10.6 ppg. Gas cut mud treated with degasser. ECD 11.1 ppg — approaching fracture gradient limit."
    },

    # DAY 10
    {
        "day": 10,
        "date": "2024-01-24",
        "depth_start": 4012,
        "depth_end": 4287,
        "bit_size": "17.5 inch",
        "bit_type": "PDC",
        "wob": 25,
        "rpm": 100,
        "rop": 35,
        "mud_weight": 10.8,
        "mud_viscosity": 64,
        "pump_pressure": 2510,
        "ecd": 11.3,
        "formation": "Deep sandstone reservoir — main target formation",
        "incidents": None,
        "npt_hours": 0,
        "operations": [
            "06:00 - Drilled into top of main target reservoir at 4,025 feet",
            "06:00 - Strong gas reading 320 units — excellent reservoir indication",
            "06:00 - Oil in mud pits confirmed by fluorescence — EXCELLENT SHOW",
            "08:00 - Reduced ROP to 15 ft/hr for careful reservoir drilling",
            "08:00 - Cuttings samples collected every 5 feet for detailed analysis",
            "10:00 - Core cut requested by company geologist — 30 feet of core recovered",
            "10:00 - Core shows excellent oil-stained sandstone, porosity estimated 24%",
            "14:00 - Continued drilling through reservoir section",
            "18:00 - LWD resistivity shows oil-water contact at approximately 4,210 feet",
            "20:00 - Drilled through oil-water contact into water leg below",
            "22:00 - Decision made to set casing and prepare for production testing",
            "24:00 - Final depth 4,287 feet. WELL OBJECTIVES ACHIEVED.",
            "24:00 - Preparing to run and cement 13-3/8 inch intermediate casing."
        ],
        "safety_notes": "Excellent day. Reservoir encountered as prognosed. All well control procedures maintained throughout.",
        "mud_report": "Final mud weight 10.8 ppg. Oil-based mud contamination noted from reservoir influx — monitoring closely."
    }
]


# ============================================================
# REPORT GENERATION FUNCTIONS
# ============================================================

def format_operations_log(operations):
    """
    Takes a list of operation strings and formats them
    into a numbered, readable operations log.
    """
    # Join all operations with newlines
    # enumerate starts counting from 1
    return "\n".join([f"  {op}" for op in operations])


def generate_report_text(day_data):
    """
    Takes one day's data dictionary and generates
    a complete, realistic DDR text document.

    This function creates the full text of one Daily Drilling Report.
    The f-string format lets us insert variables directly into the text.
    """

    # Calculate total depth drilled this day
    depth_drilled = day_data["depth_end"] - day_data["depth_start"]

    # Format the operations log
    ops_log = format_operations_log(day_data["operations"])

    # Handle incidents — if None, show "No incidents reported"
    incident_text = day_data["incidents"] if day_data["incidents"] else "No incidents reported"

    # Handle NPT
    npt_text = f"{day_data['npt_hours']} hours" if day_data['npt_hours'] > 0 else "Zero NPT — 100% productive time"

    # Build the complete report text
    # This is a multi-line f-string (text with variables inside {})
    report = f"""
================================================================================
                        DAILY DRILLING REPORT (DDR)
================================================================================

WELL INFORMATION
----------------
Well Name        : {WELL_INFO['well_name']}
Field            : {WELL_INFO['field']}
Operator         : {WELL_INFO['operator']}
Rig Name         : {WELL_INFO['rig_name']}
Location         : {WELL_INFO['location']}
Report Date      : {day_data['date']}
Report Day       : Day {day_data['day']} of operations
Prepared By      : Drilling Engineer on duty

DEPTH SUMMARY
-------------
Depth at Start of Day    : {day_data['depth_start']:,} feet
Depth at End of Day      : {day_data['depth_end']:,} feet
Total Footage Drilled    : {depth_drilled:,} feet
Target Total Depth       : {WELL_INFO['target_depth']:,} feet
Progress to Target       : {(day_data['depth_end']/WELL_INFO['target_depth']*100):.1f}%

DRILLING PARAMETERS
-------------------
Bit Size                 : {day_data['bit_size']}
Bit Type                 : {day_data['bit_type']}
Weight on Bit (WOB)      : {day_data['wob']} klbs
Rotary Speed (RPM)       : {day_data['rpm']} RPM
Rate of Penetration      : {day_data['rop']} ft/hr (average)
Pump Pressure            : {day_data['pump_pressure']} psi
Equivalent Circ. Density : {day_data['ecd']} ppg

MUD / DRILLING FLUID REPORT
----------------------------
Mud Weight               : {day_data['mud_weight']} ppg
Funnel Viscosity         : {day_data['mud_viscosity']} seconds
{day_data['mud_report']}

FORMATION DESCRIPTION
---------------------
Formation Drilled        : {day_data['formation']}

OPERATIONS LOG (24-HOUR SUMMARY)
---------------------------------
{ops_log}

INCIDENTS & NON-PRODUCTIVE TIME
---------------------------------
Incident Description     : {incident_text}
Non-Productive Time      : {npt_text}
Productive Time          : {24 - day_data['npt_hours']:.1f} hours

SAFETY REPORT
-------------
{day_data['safety_notes']}

NEXT 24-HOUR PLAN
-----------------
Continue drilling operations as per well program.
Monitor formation pressure and adjust mud weight as required.
Maintain all safety protocols and conduct pre-tour safety meetings.

================================================================================
                              END OF DAILY REPORT
                    Day {day_data['day']} — {WELL_INFO['well_name']}
================================================================================
"""
    return report


def save_report_as_txt(report_text, day_number, date):
    """
    Saves a report as a plain text file.
    Text files are easier to read than PDFs for initial testing.
    We will add PDF support in Phase 4.
    """
    # Create filename: e.g., "DDR_Day01_2024-01-15.txt"
    filename = f"DDR_Day{day_number:02d}_{date}.txt"
    # :02d means: format as integer with minimum 2 digits (01, 02, ... 10)

    filepath = os.path.join(OUTPUT_DIR, filename)

    # Write the report to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    # encoding='utf-8': Supports all characters including Arabic

    return filepath


def save_metadata(all_reports_info):
    """
    Saves metadata about all generated reports as a JSON file.
    Metadata = data about data (file names, dates, key facts).
    This helps us quickly understand what reports we have.
    """
    metadata_path = os.path.join(OUTPUT_DIR, "reports_metadata.json")

    with open(metadata_path, 'w', encoding='utf-8') as f:
        # json.dump: Convert Python dictionary to JSON format
        # indent=2: Format nicely with 2-space indentation
        json.dump(all_reports_info, f, indent=2, ensure_ascii=False)

    return metadata_path


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  DRILLING REPORT GENERATOR")
    print(f"  Well: {WELL_INFO['well_name']}")
    print("="*60 + "\n")

    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # exist_ok=True: Don't error if folder already exists

    all_reports_info = []  # Track all generated reports
    total_npt = 0          # Track total non-productive time
    total_footage = 0      # Track total footage drilled

    # Generate one report for each day
    for day_data in DAILY_DATA:

        print(f"Generating Day {day_data['day']} report ({day_data['date']})...")

        # Generate the report text
        report_text = generate_report_text(day_data)

        # Save as text file
        filepath = save_report_as_txt(
            report_text,
            day_data['day'],
            day_data['date']
        )

        # Track statistics
        footage = day_data['depth_end'] - day_data['depth_start']
        total_footage += footage
        total_npt += day_data['npt_hours']

        # Save info about this report for metadata
        all_reports_info.append({
            "day": day_data['day'],
            "date": day_data['date'],
            "filename": os.path.basename(filepath),
            "depth_start": day_data['depth_start'],
            "depth_end": day_data['depth_end'],
            "footage_drilled": footage,
            "formation": day_data['formation'],
            "incidents": day_data['incidents'],
            "npt_hours": day_data['npt_hours'],
            "has_incident": day_data['incidents'] is not None
        })

        print(f"  ✅ Saved: {filepath}")
        print(f"     Depth: {day_data['depth_start']:,} → {day_data['depth_end']:,} ft")
        if day_data['incidents']:
            print(f"     ⚠️  Incident: {day_data['incidents'][:50]}...")

    # Save metadata file
    metadata_path = save_metadata(all_reports_info)

    # Print summary
    print("\n" + "="*60)
    print("  GENERATION COMPLETE")
    print("="*60)
    print(f"  Reports generated : {len(DAILY_DATA)}")
    print(f"  Total footage     : {total_footage:,} feet")
    print(f"  Total NPT         : {total_npt:.1f} hours")
    print(f"  Days with incidents: {sum(1 for d in DAILY_DATA if d['incidents'])}")
    print(f"  Output folder     : {OUTPUT_DIR}/")
    print(f"  Metadata saved    : {metadata_path}")
    print("\n  Reports are ready for Phase 4: Preprocessing!")
    print("="*60 + "\n")

    # Print what questions our system should be able to answer
    print("  SAMPLE QUESTIONS YOUR Q&A BOT SHOULD ANSWER:")
    print("  " + "-"*48)
    questions = [
        "Was there a stuck pipe incident and on which day?",
        "What was the maximum mud weight used?",
        "How many hours of NPT were recorded on Day 6?",
        "What formation was encountered on Day 5?",
        "What was the average ROP on Day 2?",
        "Were there any oil shows? If so, when?",
        "What caused the lost circulation event?",
        "What was the final well depth achieved?",
        "Which day had a pump failure?",
        "What was the ECD on Day 9?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"  {i:2}. {q}")
    print()
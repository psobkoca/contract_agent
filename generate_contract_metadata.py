import os
import csv

METADATA_ROWS = [
    {
        "contract_id": "CTR_001",
        "contract_type": "NDA",
        "counterparty_name": "Zephyr Robotics Inc.",
        "governing_law": "Delaware",
        "effective_date": "2026-01-15",
        "contract_value_usd": 0.0,
        "our_role": "CLIENT",
        "review_priority": "MEDIUM"
    },
    {
        "contract_id": "CTR_002",
        "contract_type": "NDA",
        "counterparty_name": "Aurora Pharmaceuticals Inc.",
        "governing_law": "Massachusetts",
        "effective_date": "2026-02-12",
        "contract_value_usd": 0.0,
        "our_role": "CLIENT",
        "review_priority": "LOW"
    },
    {
        "contract_id": "CTR_003",
        "contract_type": "MSA",
        "counterparty_name": "Apex FinTech Solutions Inc.",
        "governing_law": "New York",
        "effective_date": "2026-03-01",
        "contract_value_usd": 75000.0,
        "our_role": "SERVICE_PROVIDER",
        "review_priority": "HIGH"
    },
    {
        "contract_id": "CTR_004",
        "contract_type": "MSA",
        "counterparty_name": "Vertex E-Commerce Corp",
        "governing_law": "California",
        "effective_date": "2026-03-20",
        "contract_value_usd": 120000.0,
        "our_role": "SERVICE_PROVIDER",
        "review_priority": "URGENT"
    },
    {
        "contract_id": "CTR_005",
        "contract_type": "SaaS",
        "counterparty_name": "Global Freight Logistics Corp",
        "governing_law": "Texas",
        "effective_date": "2026-04-05",
        "contract_value_usd": 60000.0,
        "our_role": "LICENSOR",
        "review_priority": "HIGH"
    },
    {
        "contract_id": "CTR_006",
        "contract_type": "SaaS",
        "counterparty_name": "TalentAcquire International Inc.",
        "governing_law": "Washington",
        "effective_date": "2026-04-25",
        "contract_value_usd": 14400.0,
        "our_role": "LICENSOR",
        "review_priority": "LOW"
    },
    {
        "contract_id": "CTR_007",
        "contract_type": "Vendor",
        "counterparty_name": "Titanium Manufacturing Partners LLC",
        "governing_law": "Ohio",
        "effective_date": "2026-05-08",
        "contract_value_usd": 150000.0,
        "our_role": "BUYER",
        "review_priority": "MEDIUM"
    },
    {
        "contract_id": "CTR_008",
        "contract_type": "Vendor",
        "counterparty_name": "EcoPack Packaging Supplies LLC",
        "governing_law": "Oregon",
        "effective_date": "2026-05-18",
        "contract_value_usd": 45000.0,
        "our_role": "BUYER",
        "review_priority": "LOW"
    },
    {
        "contract_id": "CTR_009",
        "contract_type": "Partnership",
        "counterparty_name": "Vanguard Gaming Labs Inc.",
        "governing_law": "Nevada",
        "effective_date": "2026-06-01",
        "contract_value_usd": 0.0,
        "our_role": "CLIENT",
        "review_priority": "MEDIUM"
    },
    {
        "contract_id": "CTR_010",
        "contract_type": "Partnership",
        "counterparty_name": "MetroGreen Mobility LLC",
        "governing_law": "Illinois",
        "effective_date": "2026-06-15",
        "contract_value_usd": 0.0,
        "our_role": "CLIENT",
        "review_priority": "HIGH"
    }
]

def main():
    os.makedirs("contracts", exist_ok=True)
    filepath = os.path.join("contracts", "contract_metadata.csv")
    
    fieldnames = [
        "contract_id",
        "contract_type",
        "counterparty_name",
        "governing_law",
        "effective_date",
        "contract_value_usd",
        "our_role",
        "review_priority"
    ]
    
    print(f"Generating contract metadata at {filepath}...")
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(METADATA_ROWS)
        
    print("Contract metadata CSV successfully generated!")

if __name__ == "__main__":
    main()

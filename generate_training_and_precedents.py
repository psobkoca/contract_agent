import os
import csv
import random

# 1. GENERATE TRAINING DATA
CLASSES = [
    "Confidentiality",
    "Governing Law",
    "Termination",
    "Indemnification",
    "Limitation of Liability",
    "Intellectual Property",
    "Payment Terms",
    "Force Majeure",
    "Dispute Resolution",
    "Assignment"
]

CONFIDENTIALITY_PATTERNS = [
    "The Receiving Party shall maintain all Confidential Information in strict confidence and shall not disclose it to any third party without Disclosing Party's prior written consent.",
    "The obligations of confidentiality hereunder shall survive for a period of {num} years following the termination or expiration of this Agreement.",
    "Confidential Information shall not include information that is or becomes publicly available through no breach of this Agreement by the Receiving Party.",
    "Each party agrees to protect the other's Proprietary Data with at least the same degree of care it uses for its own data of similar importance, but no less than reasonable care.",
    "The Receiving Party shall restrict access to Confidential Information to employees and contractors who have a need to know and are bound by written confidentiality obligations.",
    "Upon termination of this Agreement, the Receiving Party shall return or destroy all copies of the Disclosing Party's Confidential Information within {num} business days.",
    "Any disclosure of Confidential Information required by a court of competent jurisdiction shall not be deemed a breach, provided that prompt notice is given to the Disclosing Party.",
    "The parties acknowledge that unauthorized disclosure of Confidential Information will cause irreparable harm for which monetary damages alone would be inadequate.",
    "Confidential Information includes, without limitation, technical documentation, software code, business strategies, pricing structures, and customer records.",
    "No license or right under any patent, copyright, or trade secret is granted or implied by the disclosure of Confidential Information under this Agreement."
]

GOVERNING_LAW_PATTERNS = [
    "This Agreement shall be governed by and construed in accordance with the laws of the State of {state}, without giving effect to any principles of conflict of laws.",
    "The parties agree that any disputes arising out of this contract shall be governed by and interpreted under {state} law.",
    "This Agreement is governed by the laws of the State of {state}, excluding its conflict of laws provisions.",
    "Any legal action, suit, or proceeding arising out of or relating to this Agreement must be instituted in the state or federal courts of {state}.",
    "The United Nations Convention on Contracts for the International Sale of Goods does not apply to this Agreement.",
    "This contract shall be interpreted, construed, and enforced in all respects in accordance with the domestic laws of the State of {state}.",
    "The parties irrevocably submit to the exclusive jurisdiction and venue of the courts located in {county} County, State of {state}.",
    "Each party waives any objection it may now or hereafter have to the venue of any action brought in the courts of {state} based on forum non conveniens.",
    "The validity and construction of this Agreement shall be determined under the laws of the State of {state}.",
    "All matters arising out of or relating to this Agreement shall be governed by the laws of the State of {state}."
]

TERMINATION_PATTERNS = [
    "Either party may terminate this Agreement at any time for convenience upon {num} days' prior written notice to the other party.",
    "This Agreement may be terminated immediately by either party if the other party breaches any material term and fails to cure such breach within {num} days of receipt of written notice.",
    "Upon termination or expiration of this Agreement, all rights and licenses granted to Customer shall immediately cease and terminate.",
    "Either party may terminate this Agreement with immediate effect if the other party becomes insolvent or files a petition for bankruptcy.",
    "Termination of this Agreement shall not affect any accrued rights to payment or other liabilities incurred prior to the date of termination.",
    "The provisions of Sections {secs} shall survive any termination or expiration of this Agreement.",
    "If either party fails to perform its obligations hereunder, the non-breaching party may suspend performance or terminate the contract with {num} days' notice.",
    "This Agreement shall automatically renew for successive {num}-month periods unless either party gives notice of non-renewal at least {num} days prior to the end of the term.",
    "Upon expiration or termination, Client shall immediately pay all outstanding unpaid fees and expenses due to the Service Provider.",
    "This Agreement may be terminated by mutual written agreement of both parties at any time."
]

INDEMNIFICATION_PATTERNS = [
    "Each party shall indemnify, defend, and hold harmless the other party from and against any and all claims, liabilities, losses, and damages arising out of its material breach of this Agreement.",
    "Provider shall indemnify, defend, and hold harmless Customer against any third-party claims alleging that the Software infringes any patent, copyright, or trademark.",
    "The indemnifying party's obligations are conditioned upon the indemnified party giving prompt written notice of the claim and cooperation in the defense thereof.",
    "Customer agrees to indemnify and hold harmless Provider from any claims arising out of Customer's unauthorized use of the Service or violation of applicable laws.",
    "The indemnified party shall have the right to participate in the defense of any claim with counsel of its own choosing at its own expense.",
    "Vendor shall defend and indemnify Buyer against any claims for personal injury or property damage caused by defective Products supplied hereunder.",
    "No settlement of an indemnified claim that imposes any liability or obligation on the indemnified party shall be entered into without its prior written consent.",
    "The indemnifying party shall have sole control over the defense and settlement of any third-party claim, subject to the indemnified party's consent rights.",
    "Each Partner shall indemnify the Partnership for any losses resulting from their fraud, gross negligence, or willful misconduct.",
    "The indemnification obligations set forth in this Section shall survive the termination or expiration of this Agreement."
]

LIMITATION_OF_LIABILITY_PATTERNS = [
    "In no event shall either party be liable to the other for any indirect, special, incidental, punitive, or consequential damages arising out of or in connection with this Agreement.",
    "The maximum aggregate liability of either party under this Agreement shall not exceed the total fees paid by Customer to Provider in the {num}-month period preceding the claim.",
    "This limitation of liability shall apply regardless of the form of action, whether in contract, tort, strict liability, or otherwise, even if advised of the possibility of such damages.",
    "IN NO EVENT SHALL PROVIDER'S TOTAL LIABILITY FOR ALL CLAIMS OF ANY KIND EXCEED THE SUM OF ${amount}.",
    "The parties agree that the limitations and exclusions of liability specified in this Section represent a reasonable allocation of risk.",
    "Nothing in this Agreement shall limit or exclude either party's liability for death, personal injury, or fraud caused by its negligence.",
    "The exclusions of indirect damages under this Section shall not apply to breaches of confidentiality or indemnification obligations hereunder.",
    "Customer acknowledges that the fees charged reflect this allocation of risk, and that Provider would not enter into this Agreement without these limitations.",
    "Neither party shall be liable for any loss of profits, loss of revenue, loss of data, or cost of procurement of substitute goods or services.",
    "The limitations of liability set forth herein shall apply to the maximum extent permitted by applicable law."
]

INTELLECTUAL_PROPERTY_PATTERNS = [
    "As between the parties, Provider retains all right, title, and interest in and to the Software and Services, including all intellectual property rights therein.",
    "All deliverables, software, designs, and reports developed under this Statement of Work shall be considered works made for hire and shall be the exclusive property of Client.",
    "Except as explicitly set forth herein, neither party grants the other party any license, express or implied, to its intellectual property or proprietary technology.",
    "Customer grants Provider a non-exclusive, worldwide, royalty-free license to use Customer Data solely as necessary to provide the Services under this Agreement.",
    "Any feedback, suggestions, or ideas provided by Customer regarding the Service may be used and incorporated by Provider without restriction or compensation.",
    "Each party retains sole ownership of all pre-existing intellectual property, materials, and tools owned or developed prior to the Effective Date.",
    "Subject to the terms of this Agreement, Provider grants Customer a limited, non-exclusive, non-transferable, revocable license to use the Deliverables.",
    "Customer shall not remove, alter, or obscure any copyright, trademark, or other proprietary rights notices contained on or in the Software.",
    "The intellectual property rights in any modifications, enhancements, or derivative works of the Software shall vest immediately and exclusively in Provider.",
    "Each party agrees not to register or attempt to register any trademarks, trade names, or service marks that are confusingly similar to those of the other party."
]

PAYMENT_TERMS_PATTERNS = [
    "Customer shall pay all undisputed invoices within {num} days of the date of invoice receipt.",
    "All fees and payments under this Agreement shall be made in US Dollars via wire transfer, ACH, or credit card.",
    "Late payments shall bear interest at the rate of {num}% per month or the maximum rate permitted by law, whichever is lower.",
    "Customer is responsible for all sales, use, excise, and value-added taxes arising from the transactions contemplated under this Agreement.",
    "If Customer fails to pay any undisputed invoice when due, Provider may, upon {num} days' written notice, suspend access to the SaaS Services.",
    "Service Provider shall invoice Client on a monthly basis in arrears for all Services performed during the preceding calendar month.",
    "All out-of-pocket travel and lodging expenses incurred by Service Provider must be pre-approved in writing by Client to be eligible for reimbursement.",
    "Payment obligations are non-cancelable and all fees paid under this Agreement are non-refundable except as otherwise explicitly provided.",
    "Client shall notify Service Provider in writing of any disputed invoice amounts within {num} days of invoice receipt, detailing the basis for the dispute.",
    "The subscription fees for renewal terms shall be subject to an annual increase not to exceed {num}% over the preceding term's fees."
]

FORCE_MAJEURE_PATTERNS = [
    "Neither party shall be liable for any delay or failure in performance due to circumstances beyond its reasonable control, including acts of God, war, riot, terrorism, or embargoes.",
    "If a Force Majeure event prevents a party's performance for more than {num} days, the other party may terminate this Agreement upon written notice.",
    "The affected party shall promptly notify the other party of the occurrence of the Force Majeure event and use reasonable efforts to mitigate its effects.",
    "Force Majeure events include, without limitation, earthquakes, floods, fires, strikes, government mandates, utility failures, and internet outages.",
    "The obligations of the party affected by a Force Majeure event shall be suspended only for the duration of the Force Majeure event.",
    "This Force Majeure section shall not apply to, and shall not excuse, any payment obligations of either party under this Agreement.",
    "In the event of a Force Majeure, the parties shall cooperate in good faith to identify alternative means of performing their obligations.",
    "Failure of sub-contractors or suppliers shall not be deemed a Force Majeure unless caused by a Force Majeure event affecting the sub-contractor directly.",
    "If the Force Majeure event continues for a period exceeding {num} consecutive days, either party may terminate the affected Statement of Work without penalty.",
    "The party claiming Force Majeure must prove that it took all reasonable steps to prevent and minimize the delay or failure in performance."
]

DISPUTE_RESOLUTION_PATTERNS = [
    "Any dispute, controversy, or claim arising out of or relating to this Agreement shall be resolved through binding arbitration in {city}, {state}.",
    "Before initiating any legal action or arbitration, the parties shall attempt to resolve any dispute through good-faith mediation.",
    "The prevailing party in any legal action or arbitration arising hereunder shall be entitled to recover its reasonable attorney's fees and litigation costs.",
    "The arbitration shall be conducted by a single arbitrator in accordance with the commercial rules of the American Arbitration Association (AAA).",
    "Each party agrees that any dispute resolution proceedings will be conducted only on an individual basis and not in a class or representative action.",
    "The arbitrator's award shall be final and binding, and judgment upon the award may be entered in any court having jurisdiction thereof.",
    "The parties agree to keep the existence, content, and results of any arbitration proceeding strictly confidential.",
    "During the pendency of any dispute resolution process, both parties shall continue to perform their undisputed obligations under this Agreement.",
    "Any mediation or arbitration proceedings shall be conducted in the English language.",
    "The parties consent to personal jurisdiction in the state and federal courts located in {city}, {state} for the enforcement of any arbitration award."
]

ASSIGNMENT_PATTERNS = [
    "Neither party may assign, delegate, or transfer its rights or obligations under this Agreement without the prior written consent of the other party.",
    "This Agreement shall be binding upon and inure to the benefit of the parties and their respective permitted successors and assigns.",
    "Any attempted assignment, delegation, or transfer in violation of this Section shall be null and void ab initio.",
    "Notwithstanding the foregoing, either party may assign this Agreement in its entirety to an affiliate or in connection with a merger or sale of assets.",
    "A transfer of a controlling interest in a party (excluding public stock trades) shall be deemed an assignment requiring prior written consent.",
    "In the event of a permitted assignment, the assigning party shall remain liable for all obligations arising prior to the effective date of the assignment.",
    "The assigning party must provide written notice of any assignment to the other party within {num} days of the consummation of the transaction.",
    "No assignment shall relieve the assigning party of its obligations hereunder unless explicitly agreed in writing by the other party.",
    "This Agreement is personal to the parties, and no third-party beneficiary rights are created or implied by its terms.",
    "Consent to an assignment in one instance shall not constitute consent to any subsequent assignment or transfer."
]

CLASS_PATTERNS_MAP = {
    "Confidentiality": CONFIDENTIALITY_PATTERNS,
    "Governing Law": GOVERNING_LAW_PATTERNS,
    "Termination": TERMINATION_PATTERNS,
    "Indemnification": INDEMNIFICATION_PATTERNS,
    "Limitation of Liability": LIMITATION_OF_LIABILITY_PATTERNS,
    "Intellectual Property": INTELLECTUAL_PROPERTY_PATTERNS,
    "Payment Terms": PAYMENT_TERMS_PATTERNS,
    "Force Majeure": FORCE_MAJEURE_PATTERNS,
    "Dispute Resolution": DISPUTE_RESOLUTION_PATTERNS,
    "Assignment": ASSIGNMENT_PATTERNS
}

STATES = ["Delaware", "New York", "California", "Texas", "Massachusetts", "Illinois", "Washington", "Florida", "Ohio", "Oregon"]
COUNTIES = ["New York", "Cook", "Los Angeles", "King", "Harris", "Suffolk", "Middlesex", "Multnomah"]
CITIES = ["New York City", "Chicago", "Los Angeles", "Seattle", "Houston", "Boston", "San Francisco", "Portland"]

FICTIONAL_COMPANIES = [
    "Aegis Cyber", "Zephyr Robotics", "Obsidian BioTech", "Aurora Pharma", "CloudScale Consulting",
    "Apex FinTech", "Stellar Creative", "Vertex E-Commerce", "LogiChain Software", "Global Freight",
    "HRFlow AI", "TalentAcquire", "Titanium Partners", "HeavyGear Machinery", "EcoPack Supplies",
    "FreshFoods", "Crimson Media", "Vanguard Gaming", "SilverLine Transit", "MetroGreen Mobility",
    "Nova Tech", "Beta Corp", "Omega LLC", "Alpha Solutions", "Infinity Group",
    "Quantum Labs", "Summit Ventures", "Genesis Holdings", "Vector Industries", "Delta Systems"
]

def generate_random_clause(category):
    patterns = CLASS_PATTERNS_MAP[category]
    pattern = random.choice(patterns)
    
    # Generate random section prefix
    section_num = f"{random.randint(1, 25)}.{random.randint(1, 9)}"
    
    # Fictional companies
    party_a, party_b = random.sample(FICTIONAL_COMPANIES, 2)
    
    # Fill in standard placeholders
    num = random.choice([3, 5, 7, 10, 15, 30, 45, 60, 90, 120])
    state = random.choice(STATES)
    county = random.choice(COUNTIES)
    city = random.choice(CITIES)
    amount = random.choice(["5,000", "10,000", "50,000", "100,000", "250,000", "500,000"])
    secs = random.choice(["3, 4, and 7", "5, 8, and 12", "Confidentiality and IP Ownership", "2, 3, 4, 6, and 10"])
    rate = random.choice(["1.0", "1.5", "2.0", "0.8"])
    
    # Format the pattern itself first if it has placeholders
    formatted_pattern = pattern
    try:
        formatted_pattern = pattern.format(
            num=num,
            state=state,
            county=county,
            city=city,
            amount=amount,
            secs=secs,
            rate=rate,
            party_a=party_a,
            party_b=party_b
        )
    except KeyError:
        pass
        
    # Introduce variety by wrapping the clause text in different ways:
    wrappers = [
        "Section {section_num}. {content}",
        "Clause {section_num}: {content}",
        "Under Section {section_num}, it is agreed that: {content}",
        "In accordance with Section {section_num}, {content}",
        "{content}"
    ]
    wrapper = random.choice(wrappers)
    final_text = wrapper.format(section_num=section_num, content=formatted_pattern)
    
    # Add random company names context to some wrappers to further ensure uniqueness
    if random.random() < 0.4:
        final_text = f"Between {party_a} and {party_b}: {final_text}"
        
    return final_text

def create_clauses_training_csv():
    rows = []
    # Generate 50 balanced rows per class = 500 rows
    for category in CLASSES:
        generated = set()
        while len(generated) < 50:
            clause_text = generate_random_clause(category)
            # Remove line breaks and make sure it's clean
            clause_text = " ".join(clause_text.split())
            generated.add(clause_text)
            
        for text in generated:
            rows.append({"text": text, "label": category})
            
    # Shuffle rows to mix categories
    random.shuffle(rows)
    
    # Write to CSV
    csv_path = "clauses_training.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {csv_path} with {len(rows)} rows.")


# 2. GENERATE PRECEDENTS
# 20 Fictional precedents
PRECEDENT_DETAILS = [
    {
        "id": "PRC_001",
        "title": "Standard Mutual Non-Disclosure Agreement (NDA) Template",
        "type": "Clause Template",
        "category": "Confidentiality",
        "summary": "Fictional boilerplate document containing mutual confidentiality clauses, carve-outs, return of materials, and standard remedies.",
        "text": """MUTUAL NDA TEMPLATE (STANDARD)
DOCUMENT ID: PRC_001
TYPE: Clause Template
CATEGORY: Confidentiality

This document contains standard boilerplate text for a Mutual Non-Disclosure Agreement.

CLAUSE A: DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any information disclosed by either party to the other party that is designated as confidential, or which reasonably should be understood to be confidential given the nature of the information.

CLAUSE B: NON-DISCLOSURE OBLIGATIONS
The Receiving Party shall keep the Disclosing Party's Confidential Information strictly confidential and shall not disclose it to any third party without prior written consent, except as required by law.

CLAUSE C: STANDARD TERM
The obligations under this Agreement shall survive for a period of five (5) years from the date of disclosure.
"""
    },
    {
        "id": "PRC_002",
        "title": "Governing Law and Forum Selection Guidelines",
        "type": "Regulatory Guidance",
        "category": "Governing Law",
        "summary": "Regulatory and risk assessment guidelines for selecting governing law jurisdictions and forum venue options.",
        "text": """GOVERNING LAW GUIDANCE
DOCUMENT ID: PRC_002
TYPE: Regulatory Guidance
CATEGORY: Governing Law

Selecting appropriate governing law and forum selection clauses reduces jurisdictional risks.

1. DELAWARE JURISDICTION
Delaware remains the gold standard for corporate litigation due to the specialized Court of Chancery.
Recommended Clause: "This Agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware."

2. NEW YORK JURISDICTION
Preferred for financial transactions. Highly predictable commercial division courts.
Recommended Clause: "This Agreement shall be governed by the laws of the State of New York."
"""
    },
    {
        "id": "PRC_003",
        "title": "Limitation of Liability Negotiation Playbook",
        "type": "Negotiation Playbook",
        "category": "Limitation of Liability",
        "summary": "Negotiation rules and fallback positions for limiting aggregate and consequential damages liability.",
        "text": """LIMITATION OF LIABILITY PLAYBOOK
DOCUMENT ID: PRC_003
TYPE: Negotiation Playbook
CATEGORY: Limitation of Liability

NEGO-RULES:
Rule 1: Always seek to exclude consequential, incidental, and indirect damages.
Rule 2: Cap total liability at 12 months fees paid.
Rule 3: Avoid unlimited liability caps unless required for data security or gross negligence.

FALLBACK POSITIONS:
- Fallback 1: Cap at 2x annual contract value.
- Fallback 2: Cap at a fixed sum of $500,000 or insurance coverage limits.
"""
    },
    {
        "id": "PRC_004",
        "title": "GDPR Compliance and Data Privacy Playbook",
        "type": "Regulatory Guidance",
        "category": "Confidentiality",
        "summary": "Regulatory playbook outlining compliance requirements for data processing, data transfers, and standard contractual clauses.",
        "text": """GDPR COMPLIANCE PLAYBOOK
DOCUMENT ID: PRC_004
TYPE: Regulatory Guidance
CATEGORY: Confidentiality

This document outlines regulatory guidelines for standard data processing.

SECTION 1: DATA CONTROLLER TO DATA PROCESSOR TRANSFERS
Under GDPR Article 28, a written contract is required. The contract must contain standard clauses detailing:
- The subject matter and duration of the processing.
- The nature and purpose of the processing.
- The types of personal data and categories of data subjects.

SECTION 2: DATA SUB-PROCESSING
Processors must obtain prior written consent from controllers before appointing sub-processors.
"""
    },
    {
        "id": "PRC_005",
        "title": "Master Services Agreement (MSA) Boilerplate",
        "type": "Clause Template",
        "category": "Payment Terms",
        "summary": "Standard boilerplate containing general payment terms, net payment days, and invoice dispute mechanisms.",
        "text": """MASTER SERVICES AGREEMENT TEMPLATE
DOCUMENT ID: PRC_005
TYPE: Clause Template
CATEGORY: Payment Terms

SECTION 3: PAYMENT AND BILLING
3.1 Invoices. Service Provider shall invoice Client monthly. Client shall pay all undisputed invoice amounts within thirty (30) days of receipt (Net 30).
3.2 Late Payments. Late payments shall bear interest at 1.5% per month or the highest legal rate.
3.3 Billing Disputes. Client must notify Service Provider in writing of any billing disputes within fifteen (15) days of invoice receipt.
"""
    },
    {
        "id": "PRC_006",
        "title": "Payment Term Negotiation Strategy Guide",
        "type": "Negotiation Playbook",
        "category": "Payment Terms",
        "summary": "Strategic instructions for negotiating client payment terms, handling net 30/45/60, and setting interest on late payments.",
        "text": """PAYMENT TERMS NEGOTIATION GUIDE
DOCUMENT ID: PRC_006
TYPE: Negotiation Playbook
CATEGORY: Payment Terms

STRATEGY OUTLINE:
- Target Position: Net 30 payment terms, 1.5% monthly late fee, and right to suspend services after 10 days overdue.
- Alternate Position 1: Net 45 payment terms, 1.0% late fee, right to suspend after 30 days overdue.
- Walkaway Point: Net 90 payment terms with no late fees or suspension rights.
"""
    },
    {
        "id": "PRC_007",
        "title": "Intellectual Property Transfer and Assignment Agreement",
        "type": "Clause Template",
        "category": "Intellectual Property",
        "summary": "Standard template for assigning intellectual property rights from independent contractors to a hiring company.",
        "text": """IP TRANSFER AND ASSIGNMENT TEMPLATE
DOCUMENT ID: PRC_007
TYPE: Clause Template
CATEGORY: Intellectual Property

SECTION 1: IP ASSIGNMENT
Contractor hereby irrevocably assigns and transfers to Company all right, title, and interest in and to all deliverables, software, designs, and patentable ideas developed in connection with this contract.

SECTION 2: WORKS MADE FOR HIRE
All work products created under this Agreement shall be considered "works made for hire" under US Copyright law. To the extent they do not qualify, this Agreement serves as an assignment of all such rights.
"""
    },
    {
        "id": "PRC_008",
        "title": "Force Majeure Checklist and Draft Clause",
        "type": "Negotiation Playbook",
        "category": "Force Majeure",
        "summary": "Negotiation checklist for force majeure events, exclusions, and termination thresholds.",
        "text": """FORCE MAJEURE NEGOTIATION CHECKLIST
DOCUMENT ID: PRC_008
TYPE: Negotiation Playbook
CATEGORY: Force Majeure

CHECKLIST ITEMS:
- Does the clause explicitly list pandemics, epidemics, and government shutdowns?
- Are payment obligations carved out from force majeure relief?
- Is there a termination threshold (e.g., 60 days of continuous force majeure)?
- Must the affected party provide immediate written notice (within 48 hours)?

DRAFT CLAUSE:
"Neither party shall be liable for delays caused by acts of God, war, riot, or pandemics, provided that payment obligations are not excused."
"""
    },
    {
        "id": "PRC_009",
        "title": "Dispute Resolution and Arbitration Procedures",
        "type": "Regulatory Guidance",
        "category": "Dispute Resolution",
        "summary": "Regulatory and procedural guidelines for executing binding commercial arbitration under AAA or JAMS rules.",
        "text": """DISPUTE RESOLUTION GUIDANCE DOCUMENT
DOCUMENT ID: PRC_009
TYPE: Regulatory Guidance
CATEGORY: Dispute Resolution

OVERVIEW OF ARBITRATION:
Arbitration is private dispute resolution. JAMS and AAA are the most common providers.

PROCEDURE:
1. Filing: The claimant files a demand for arbitration.
2. Selection: The parties select a neutral arbitrator from the provider's panel.
3. Discovery: Limited exchange of documents and depositions.
4. Hearing: The arbitrator hears arguments and issues a binding award.
5. Enforcement: The award is enforced in federal or state court.
"""
    },
    {
        "id": "PRC_010",
        "title": "Indemnification Risk Allocation Guide",
        "type": "Clause Template",
        "category": "Indemnification",
        "summary": "Mutual indemnification boilerplate clauses for third-party intellectual property and general commercial claims.",
        "text": """INDEMNIFICATION ALLOCATION BOILERPLATE
DOCUMENT ID: PRC_010
TYPE: Clause Template
CATEGORY: Indemnification

SECTION 8: MUTUAL INDEMNIFICATION
8.1 Provider Indemnity. Provider shall defend, indemnify, and hold harmless Client from any third-party claims alleging that the software infringes any patent or copyright.
8.2 Client Indemnity. Client shall defend, indemnify, and hold harmless Provider from any claims arising out of Client's breach of law or customer data infringement.
8.3 Conditions. Indemnification is conditioned on prompt notice and sole control of the defense.
"""
    },
    {
        "id": "PRC_011",
        "title": "SaaS Service Level Agreement (SLA) Template",
        "type": "Clause Template",
        "category": "Termination",
        "summary": "Standard SaaS SLA boilerplate with uptime metrics, service credit formulas, and termination rights for chronic SLA failures.",
        "text": """SAAS SERVICE LEVEL AGREEMENT (SLA)
DOCUMENT ID: PRC_011
TYPE: Clause Template
CATEGORY: Termination

SECTION 1: SERVICE AVAILABILITY
Provider warrants that the Service shall have an uptime availability of at least 99.9% in any calendar month.

SECTION 2: SLA CREDITS
If uptime falls below 99.9%, Customer is entitled to service credits as follows:
- Uptime 99.0% - 99.8%: 10% credit of monthly fee.
- Uptime < 99.0%: 25% credit of monthly fee and right to terminate for cause.
"""
    },
    {
        "id": "PRC_012",
        "title": "CCPA / CPRA Consumer Privacy Compliance Guide",
        "type": "Regulatory Guidance",
        "category": "Confidentiality",
        "summary": "Regulatory guidance detailing requirements under the California Consumer Privacy Act and California Privacy Rights Act.",
        "text": """CCPA COMPLIANCE GUIDELINES
DOCUMENT ID: PRC_012
TYPE: Regulatory Guidance
CATEGORY: Confidentiality

This guide details California privacy compliance standards.

SECTION 1: CONSUMER RIGHTS
Consumers have the right to know, delete, and correct personal information collected by businesses.
Businesses must provide a "Do Not Sell or Share My Personal Information" link.

SECTION 2: SERVICE PROVIDER CONTRACTS
Contracts with service providers must restrict them from retaining, using, or disclosing personal data for any purpose other than performing the services.
"""
    },
    {
        "id": "PRC_013",
        "title": "Termination for Convenience Playbook",
        "type": "Negotiation Playbook",
        "category": "Termination",
        "summary": "Negotiation manual for handling termination for convenience, wind-down fees, and transitional services.",
        "text": """TERMINATION FOR CONVENIENCE PLAYBOOK
DOCUMENT ID: PRC_013
TYPE: Negotiation Playbook
CATEGORY: Termination

NEGOTIATION PROTOCOLS:
- Target: No termination for convenience for either party during the initial term.
- Fallback 1: Unilateral right for customer only, upon 90 days notice and payment of a wind-down fee equal to 3 months of fees.
- Fallback 2: Mutual right to terminate upon 180 days notice, with transition support billed at standard consulting rates.
"""
    },
    {
        "id": "PRC_014",
        "title": "Assignment and Change of Control Playbook",
        "type": "Negotiation Playbook",
        "category": "Assignment",
        "summary": "Negotiation strategies for assignment restrictions, mergers, acquisitions, and change of control clauses.",
        "text": """ASSIGNMENT & CHANGE OF CONTROL PLAYBOOK
DOCUMENT ID: PRC_014
TYPE: Negotiation Playbook
CATEGORY: Assignment

NEGO-RULES:
Rule 1: Always allow assignment to affiliates or successors in a merger without consent.
Rule 2: Restrict assignment to direct competitors of the non-assigning party.
Rule 3: Ensure notice is given within 30 days of the assignment.

FALLBACK CLAUSES:
"Neither party may assign this Agreement without consent, except that either party may assign to a successor in a merger, acquisition, or sale of assets."
"""
    },
    {
        "id": "PRC_015",
        "title": "HIPAA Business Associate Agreement (BAA) Template",
        "type": "Clause Template",
        "category": "Confidentiality",
        "summary": "Standard HIPAA BAA boilerplate for managing protected health information (PHI) between covered entities and business associates.",
        "text": """HIPAA BUSINESS ASSOCIATE AGREEMENT (BAA)
DOCUMENT ID: PRC_015
TYPE: Clause Template
CATEGORY: Confidentiality

This template is for Business Associate relationship management.

SECTION 2: PERMITTED USES AND DISCLOSURES
Business Associate may use or disclose PHI only to perform the services specified in the Service Agreement, provided that such use or disclosure would not violate the Privacy Rule if done by Covered Entity.

SECTION 5: BREACH NOTIFICATION
Business Associate shall notify Covered Entity within twenty-four (24) hours of discovering any breach of unsecured PHI.
"""
    },
    {
        "id": "PRC_016",
        "title": "Software End-User License Agreement (EULA) Template",
        "type": "Clause Template",
        "category": "Intellectual Property",
        "summary": "Standard license grant template containing restrictions on reverse engineering, hosting, and redistribution.",
        "text": """SOFTWARE EULA TEMPLATE
DOCUMENT ID: PRC_016
TYPE: Clause Template
CATEGORY: Intellectual Property

SECTION 1: LICENSE GRANT
Licensor grants licensee a non-exclusive, non-transferable, limited license to install and run the Software solely for internal business use.

SECTION 2: LICENSE RESTRICTIONS
Licensee shall not:
- Reverse engineer, decompile, or disassemble the Software.
- Copy, modify, or create derivative works.
- Rent, lease, or distribute the Software to third parties.
"""
    },
    {
        "id": "PRC_017",
        "title": "EU Standard Contractual Clauses (SCCs) Guidance",
        "type": "Regulatory Guidance",
        "category": "Confidentiality",
        "summary": "Regulatory guide for implementing the European Commission's standard contractual clauses for international data transfers.",
        "text": """EU SCC COMPLIANCE GUIDE
DOCUMENT ID: PRC_017
TYPE: Regulatory Guidance
CATEGORY: Confidentiality

This guide addresses EU standard contractual clauses.

SECTION 1: BACKGROUND
SCCs are standard contracts approved by the European Commission to provide adequate safeguards for data transfers from the EEA to third countries.

SECTION 2: MODULE SELECTION
- Module 1: Controller-to-Controller transfers.
- Module 2: Controller-to-Processor transfers.
- Module 3: Processor-to-Processor transfers.
- Module 4: Processor-to-Controller transfers.
"""
    },
    {
        "id": "PRC_018",
        "title": "Audit and Financial Review Rights Clause",
        "type": "Clause Template",
        "category": "Payment Terms",
        "summary": "Standard clause template specifying audit rights, financial record retention, and payment discrepancy remedies.",
        "text": """AUDIT RIGHTS BOILERPLATE CLAUSE
DOCUMENT ID: PRC_018
TYPE: Clause Template
CATEGORY: Payment Terms

SECTION 12: AUDIT RIGHTS
12.1 Audits. Client shall have the right, once per year and upon 30 days notice, to audit Vendor's financial records related to services performed.
12.2 Cost. The audit shall be at Client's expense, unless the audit reveals an overcharge of more than five percent (5%), in which case Vendor shall reimburse the audit costs.
12.3 Retention. Vendor shall keep records for three (3) years.
"""
    },
    {
        "id": "PRC_019",
        "title": "Export Control and Sanctions Compliance Playbook",
        "type": "Negotiation Playbook",
        "category": "Governing Law",
        "summary": "Negotiation playbook mapping compliance requirements for US, EU, and international export controls and trade sanctions.",
        "text": """EXPORT CONTROL & SANCTIONS PLAYBOOK
DOCUMENT ID: PRC_019
TYPE: Negotiation Playbook
CATEGORY: Governing Law

NEGO-RULES:
Rule 1: Always include a warranty that neither party is on any OFAC sanctions lists.
Rule 2: Restrict the export or re-export of software to embargoed countries.
Rule 3: Ensure governing law includes compliance with the Export Administration Regulations (EAR).

FALLBACK CLAUSE:
"Each party warrants that it is in compliance with all applicable export controls, import regulations, and trade sanctions laws."
"""
    },
    {
        "id": "PRC_020",
        "title": "Warranty and Disclaimers Boilerplate Clause",
        "type": "Clause Template",
        "category": "Indemnification",
        "summary": "Standard warranty disclaimer template containing exclusions for merchantability and fitness for a particular purpose.",
        "text": """WARRANTY DISCLAIMER TEMPLATE
DOCUMENT ID: PRC_020
TYPE: Clause Template
CATEGORY: Indemnification

SECTION 10: WARRANTIES AND DISCLAIMERS
10.1 Limited Warranty. Provider warrants that the service will perform substantially in accordance with the documentation.
10.2 Disclaimer. EXCEPT AS EXPLICITLY PROVIDED HEREIN, PROVIDER DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
"""
    }
]

def generate_precedents_and_index():
    directory = "legal_precedents"
    os.makedirs(directory, exist_ok=True)
    
    index_rows = []
    
    for prec in PRECEDENT_DETAILS:
        filename = f"{prec['id']}.txt"
        filepath = os.path.join(directory, filename)
        
        # Write TXT file
        with open(filepath, mode="w", encoding="utf-8") as f:
            f.write(prec["text"])
        print(f"Generated Precedent TXT: {filepath}")
        
        # Add to index metadata
        index_rows.append({
            "file_name": filename,
            "title": prec["title"],
            "document_type": prec["type"],
            "category": prec["category"],
            "description": prec["summary"]
        })
        
    # Write precedent_index.csv
    csv_path = "precedent_index.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "title", "document_type", "category", "description"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Generated {csv_path} with {len(index_rows)} rows.")

def main():
    create_clauses_training_csv()
    generate_precedents_and_index()

if __name__ == "__main__":
    main()

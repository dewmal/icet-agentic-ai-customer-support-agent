from crewai import Task

from dotenv import load_dotenv
from crewai import Agent, LLM
import os

load_dotenv()


# ---------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------

llm = LLM(
    model="openrouter/nvidia/nemotron-3.5-lightning:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)


# =========================================================
# 1. Customer Support AI Agent (CSA)
# =========================================================

customer_support_agent = Agent(
    role="Customer Support AI Agent",

    goal="""
    Improve the First-Time Fix Rate by collecting complete and accurate
    diagnostic information before a technician visit.

    Verify warranty eligibility before dispatch, clearly explain warranty
    coverage and possible customer charges, manage service waitlists during
    peak demand, communicate realistic service windows, and keep customers
    proactively informed while they wait for service.

    Ensure every service case contains sufficient information for the
    Service Coordinator and Field Technician to act without unnecessary
    follow-up.
    """,

    backstory="""
    You are an experienced appliance service customer-support specialist.

    You handle incoming repair and service requests and are highly skilled
    at structured troubleshooting, warranty verification, customer
    communication, and service-ticket preparation.

    When handling a service request, you gather all relevant information,
    including:

    - Customer contact and service-location details
    - Appliance or product type
    - Brand
    - Model number
    - Serial number
    - Purchase date
    - Proof of purchase
    - Detailed description of the problem
    - Error codes
    - Unusual noises, smells, leaks, vibrations, or behavior
    - When the problem started
    - Whether the problem is intermittent or continuous
    - Troubleshooting steps already attempted
    - Photos or short videos when useful
    - Photos of the product identification label when required

    You use standardized troubleshooting questions appropriate to the
    appliance or product category.

    You carefully distinguish between mechanical or electrical failures and
    physical, accidental, cosmetic, misuse-related, or externally caused
    damage.

    You never assume that a fault is covered by warranty. Warranty status
    must be verified using available warranty information, proof of purchase,
    serial numbers, model numbers, and applicable warranty policies.

    When discussing warranty, you clearly communicate:

    - Whether the warranty appears active
    - Whether the reported type of fault appears covered
    - Any exclusions or limitations
    - Any diagnostic, inspection, labor, transportation, or repair fees that
      may apply
    - That final coverage may depend on technician inspection where required

    If a diagnostic fee such as $50 applies according to the relevant
    policy, you disclose it clearly before scheduling.

    You document all findings accurately in the service record and assign an
    initial diagnostic category based on the information available.

    If scheduling capacity is limited, you do not promise unavailable dates.
    You communicate the current backlog, provide a realistic estimated
    service window, and place the customer on the appropriate structured
    waitlist when necessary.

    Customers waiting for initial service should receive proactive status
    updates every three days.

    Before a case is considered ready for scheduling or routing to the
    Service Coordinator, verify that:

    1. Required customer and product information has been collected.
    2. Symptoms and diagnostic information are documented.
    3. Warranty status has been checked.
    4. Coverage limitations and possible charges have been explained.
    5. The customer's agreement or acknowledgement is documented.

    Your communication is professional, concise, transparent, empathetic,
    and operationally accurate. You do not fabricate warranty information,
    appointment availability, technician availability, prices, diagnoses,
    parts availability, or service commitments.
    """,

    llm=llm,
    verbose=True,
    allow_delegation=False
)


# =========================================================
# 2. Service Coordinator AI Agent (SC)
# =========================================================

service_coordinator_agent = Agent(
    role="Service Coordinator AI Agent",

    goal="""
    Coordinate service operations efficiently by managing technician
    capacity, scheduling, routing, parts availability, and supply-chain
    delays.

    Minimize technician travel time, prevent overbooking, assign work based
    on technician capability and geographic proximity, and keep customers
    informed when service is delayed because required parts are unavailable.
    """,

    backstory="""
    You are an experienced appliance-service operations coordinator
    specializing in dispatch, field-service scheduling, parts management,
    capacity planning, and customer delay management.

    You receive service cases that have already been reviewed and prepared
    by the Customer Support AI Agent.

    Before scheduling or dispatching a job, you review the service-ticket
    information, including:

    - Customer location
    - Appliance or product type
    - Initial diagnostic category
    - Symptoms and error codes
    - Required technician skills
    - Possible required parts
    - Possible specialized tools
    - Warranty and service information
    - Scheduling constraints

    You use scheduling and calendar information to determine available
    capacity.

    You strictly respect the configured maximum number of jobs that can be
    performed each day. You do not overbook technicians merely to satisfy
    customer demand.

    During seasonal peaks or periods of high demand, you maintain capacity
    limits and use structured scheduling or waitlist procedures rather than
    creating unrealistic appointment commitments.

    When assigning jobs, you consider:

    - Technician skill set
    - Appliance or product expertise
    - Technician availability
    - Customer location
    - Geographic proximity between jobs
    - Existing route schedule
    - Required parts
    - Required tools
    - Daily capacity limits

    Your objective is to create efficient daily routes while ensuring the
    technician assigned to each case has the appropriate capability.

    You also manage parts-related service delays.

    When a required part is not available in standard stock, you:

    1. Check the parts catalog or supplier information.
    2. Determine current availability.
    3. Determine the supplier's estimated lead time or ETA.
    4. Initiate the appropriate parts-ordering process when authorized.
    5. Place the service case in a "Waiting on Parts" status or queue.
    6. Record the part information and ETA in the service record.
    7. Communicate the delay clearly.
    8. Ensure the customer receives an update at least weekly until the part
       arrives.

    You never invent supplier availability, lead times, appointment slots,
    technician capacity, or part ETAs. If the required operational data is
    unavailable, you explicitly state what must be checked.

    Before finalizing a daily job schedule, verify that:

    - Every assigned technician has the required skill set.
    - Jobs are geographically sensible where practical.
    - Required known parts or tools have been considered.
    - The total number of assigned jobs does not exceed configured daily
      capacity.

    Your decisions prioritize operational reliability over unrealistic
    scheduling promises.
    """,

    llm=llm,
    verbose=True,
    allow_delegation=False
)


# =========================================================
# 3. Field Technician AI Assistant (FT)
# =========================================================

field_technician_agent = Agent(
    role="Field Technician AI Assistant",

    goal="""
    Help field technicians arrive prepared, diagnose faults systematically,
    maximize the probability of completing repairs during the first visit,
    and consistently apply repair-versus-replacement rules.

    Identify likely tools and spare parts before dispatch, support structured
    on-site diagnosis, and recommend replacement rather than repair when the
    defined cost-benefit criteria indicate replacement is more appropriate.
    """,

    backstory="""
    You are an expert appliance field-service assistant supporting
    technicians before and during customer visits.

    Your primary objective is to improve the First-Time Fix Rate while
    ensuring repair decisions are technically and economically justified.

    Before dispatch, you carefully review the assigned service ticket,
    including:

    - Appliance or product type
    - Brand and model
    - Serial number when available
    - Customer-reported symptoms
    - Error codes
    - Photos or videos
    - Previous troubleshooting results
    - Initial diagnostic category
    - Previous repair history when available
    - Warranty information
    - Suspected components
    - Technician notes
    - Known required tools
    - Known required spare parts

    Based on the ticket information, you identify likely diagnostic tools,
    specialized tools, commonly required components, and appropriate
    standard-stock spare parts that should be loaded before departure.

    You do not claim that a component is definitely defective solely from
    customer-reported symptoms. Pre-visit assessments are treated as likely
    causes or diagnostic hypotheses until confirmed through proper testing.

    When the technician begins travel to the customer's location, the job
    should be marked "En Route" according to the service workflow.

    You recognize that changing the status should trigger the configured
    automated customer notification, and you verify that the notification
    mechanism has successfully executed when such system information is
    available.

    On site, you support systematic diagnosis rather than unnecessary parts
    replacement.

    Diagnostic guidance should consider:

    - Visual inspection
    - Error-code interpretation
    - Mechanical inspection
    - Electrical measurements where appropriate
    - Wiring and connector inspection
    - Sensor checks
    - Component testing
    - Operating conditions
    - Customer usage information
    - Safety considerations
    - Manufacturer service procedures when available

    Repair-versus-replacement decisions must strictly follow the provided
    Repair-vs-Replacement Decision Matrix.

    In particular, a replacement option should be presented when:

    - The estimated repair cost exceeds 50% of the cost of an equivalent
      replacement unit, OR
    - The appliance or product is beyond its defined expected service life.

    You do not arbitrarily override these thresholds.

    When repair remains economically viable but requires a part that is not
    available in standard stock, you prepare a clear part request for the
    Service Coordinator containing, where available:

    - Part name
    - Part number
    - Appliance model
    - Appliance serial number
    - Quantity
    - Diagnostic reason
    - Urgency
    - Relevant technician notes

    You never fabricate test results, part numbers, measurements, repair
    prices, replacement prices, inventory quantities, or manufacturer
    instructions.

    If critical information is unavailable, identify what must be checked
    rather than guessing.

    Your recommendations should be technically precise, practical for field
    technicians, safety-conscious, and focused on completing viable repairs
    correctly on the first visit.
    """,

    llm=llm,
    verbose=True,
    allow_delegation=False
)
# =========================================================
# SOP 01
# Pre-Dispatch Customer Triage and Diagnostic Assessment
# Responsible Agent: CSA
# =========================================================

pre_dispatch_triage_task = Task(
    description="""
    Review the incoming customer service or repair request.

    Customer request:
    {customer_message}

    Perform the Pre-Dispatch Customer Triage and Diagnostic Assessment
    according to SOP 01.

    Your responsibilities are to:

    1. Identify the customer-reported appliance or product issue.
    2. Determine what information has already been provided.
    3. Identify any missing information required for diagnosis, including:
       - Product/appliance type
       - Brand
       - Model number
       - Serial number
       - Description of the fault
       - Error codes
       - Unusual noises
       - Leaks
       - Smells
       - Vibrations
       - When the problem started
       - Whether the fault is intermittent or continuous
       - Troubleshooting already attempted
    4. Ask targeted troubleshooting questions relevant to the reported issue.
    5. Request a photo or short video when visual evidence would improve
       diagnosis.
    6. Request a photo of the appliance/product identification label when
       model or serial information is missing or unclear.
    7. Record all known symptoms, error codes, troubleshooting results,
       and available visual evidence.
    8. Assign an initial diagnostic category based only on available
       evidence.
    9. Identify likely diagnostic areas for the field technician without
       claiming that an unverified component has definitely failed.
    10. Determine whether the case contains enough diagnostic information
        to continue to warranty verification and scheduling.

    Do not invent missing customer information, diagnostic results,
    model numbers, serial numbers, photos, or error codes.

    If essential information is missing, clearly state what must be
    collected from the customer.
    """,

    expected_output="""
    A structured Pre-Dispatch Triage Report containing:

    - Customer issue summary
    - Product/appliance details available
    - Reported symptoms
    - Error codes
    - Troubleshooting already attempted
    - Photos/videos received or required
    - Missing diagnostic information
    - Follow-up questions for the customer
    - Initial diagnostic category
    - Likely diagnostic areas
    - Triage completeness status:
        COMPLETE
        or
        CUSTOMER INFORMATION REQUIRED
    - Recommended next action
    """,

    agent=customer_support_agent
)


# =========================================================
# SOP 03
# Warranty Claim Verification and Expectation Management
# Responsible Agent: CSA
# =========================================================

warranty_verification_task = Task(
    description="""
    Perform Warranty Claim Verification and Expectation Management
    according to SOP 03.

    Use the available customer and product information from the service
    case.

    Customer/service information:
    {customer_message}

    Warranty information if available:
    {warranty_information}

    Warranty policy if available:
    {warranty_policy}

    Your responsibilities are to:

    1. Verify whether the following information is available:
       - Product serial number
       - Product model number
       - Proof of purchase
       - Purchase date
    2. Determine whether warranty eligibility can actually be verified
       using the supplied warranty information.
    3. Determine whether the reported fault appears to fall within
       warranty coverage.
    4. Distinguish mechanical/electrical failure from physical,
       accidental, cosmetic, misuse-related, or other excluded damage.
    5. Identify warranty inclusions and exclusions.
    6. Clearly identify any possible customer charges such as:
       - Diagnostic fees
       - Labor charges
       - Parts charges
       - Transportation/service-call charges
    7. Clearly explain the warranty position and possible costs to the
       customer.
    8. Record whether customer acknowledgement or agreement has been
       received.
    9. Determine whether the case is cleared to proceed to scheduling.

    Never assume warranty coverage.

    If warranty database information, warranty policy information,
    proof of purchase, serial number, or another required input is not
    available, state that verification is pending rather than fabricating
    a warranty decision.
    """,

    expected_output="""
    A structured Warranty Verification Report containing:

    - Model number
    - Serial number
    - Purchase information
    - Proof-of-purchase status
    - Warranty verification status
    - Warranty period status
    - Reported fault classification
    - Covered items
    - Excluded items
    - Potential customer charges
    - Customer-facing warranty explanation
    - Customer acknowledgement status
    - Scheduling clearance:
        CLEARED
        NOT CLEARED
        or
        VERIFICATION REQUIRED
    - Required next action
    """,

    agent=customer_support_agent,

    context=[pre_dispatch_triage_task]
)


# =========================================================
# SOP 04A
# Peak Season Capacity and Backlog Assessment
# Responsible Agent: SC
# =========================================================

capacity_and_backlog_task = Task(
    description="""
    Perform the Service Coordinator portion of SOP 04:
    Peak Season Scheduling and Backlog Communication Protocols.

    Scheduling/capacity information:
    {scheduling_information}

    Technician availability:
    {technician_availability}

    Determine current field-service capacity.

    Your responsibilities are to:

    1. Determine how many technicians are available.
    2. Review configured daily scheduling capacity.
    3. Determine the maximum number of service jobs that can safely be
       scheduled.
    4. Identify whether current demand exceeds available capacity.
    5. Prevent recommendations that would cause overbooking.
    6. Determine the realistic current service window.
    7. Determine whether customers should be:
       - Directly scheduled
       - Offered a scheduling window
       - Placed on a structured waitlist
    8. Identify any relevant service bottlenecks.

    Do not invent technician availability, appointment slots, backlog
    size, or scheduling capacity.

    If live scheduling information is unavailable, explicitly identify
    the information that must be checked.
    """,

    expected_output="""
    A Capacity and Backlog Assessment containing:

    - Available technician count
    - Maximum daily job capacity
    - Current scheduled workload
    - Remaining capacity
    - Backlog status
    - Estimated current service window
    - Overbooking risk
    - Scheduling recommendation
    - Waitlist requirement
    - Capacity verification status
    """,

    agent=service_coordinator_agent
)


# =========================================================
# SOP 04B
# Customer Scheduling and Backlog Communication
# Responsible Agent: CSA
# =========================================================

backlog_customer_communication_task = Task(
    description="""
    Perform the Customer Support Agent portion of SOP 04:
    Peak Season Scheduling and Backlog Communication Protocols.

    Use the Service Coordinator's capacity and backlog assessment together
    with the verified service case.

    Your responsibilities are to:

    1. Clearly communicate the current service backlog to the customer.
    2. Provide the realistic estimated service window determined by the
       Service Coordinator.
    3. Do not promise a specific date unless it is confirmed as available.
    4. Determine whether the customer should be:
       - Scheduled
       - Offered an available service window
       - Added to the structured waitlist
    5. Prepare a confirmation message explaining:
       - Current service status
       - Estimated wait time
       - Scheduling or waitlist status
       - Emergency escalation process where applicable
    6. Establish the requirement for proactive updates every 3 days while
       the customer remains waiting.
    7. Record the customer's scheduling acknowledgement.

    Do not invent appointment availability.

    Do not state that the customer has been scheduled unless confirmed
    scheduling information exists.
    """,

    expected_output="""
    A Customer Scheduling Communication Record containing:

    - Current backlog explanation
    - Estimated service window
    - Scheduling status
    - Waitlist status
    - Customer confirmation message
    - Emergency escalation guidance if applicable
    - Next customer update due
    - Customer acknowledgement status
    - Recommended next action
    """,

    agent=customer_support_agent,

    context=[
        warranty_verification_task,
        capacity_and_backlog_task
    ]
)


# =========================================================
# SOP 05A
# Daily Route Scheduling and Technician Assignment
# Responsible Agent: SC
# =========================================================

technician_dispatch_planning_task = Task(
    description="""
    Perform the Service Coordinator portion of SOP 05:
    Field Technician Dispatch and Inventory Preparation.

    Daily scheduled service jobs:
    {daily_jobs}

    Technician information:
    {technician_availability}

    Your responsibilities are to:

    1. Review all scheduled service jobs for the day.
    2. Review technician availability and technician skill sets.
    3. Match each service job to a technician with the required skills.
    4. Consider customer geographical locations.
    5. Organize jobs to reduce unnecessary technician travel.
    6. Consider diagnostic categories and known repair requirements.
    7. Ensure the total number of assigned jobs does not exceed maximum
       daily capacity.
    8. Produce a logical daily route/job manifest.
    9. Highlight jobs requiring unusual tools, special skills, or likely
       spare parts.

    Do not invent technician skills, geographic information, capacity,
    or job availability.

    If required information is unavailable, clearly identify the missing
    operational data.
    """,

    expected_output="""
    A Daily Technician Dispatch Plan containing:

    - Technician assignments
    - Assigned service tickets
    - Technician skill match
    - Proposed job sequence
    - Geographic/routing considerations
    - Special service requirements
    - Possible tools/parts requirements
    - Jobs per technician
    - Total jobs scheduled
    - Maximum daily capacity
    - Capacity validation:
        WITHIN CAPACITY
        or
        CAPACITY ISSUE
    """,

    agent=service_coordinator_agent,

    context=[capacity_and_backlog_task]
)


# =========================================================
# SOP 05B
# Technician Pre-Dispatch Inventory Preparation
# Responsible Agent: FT
# =========================================================

inventory_preparation_task = Task(
    description="""
    Perform the Field Technician portion of SOP 05:
    Field Technician Dispatch and Inventory Preparation.

    Standard inventory information:
    {inventory_information}

    Review the assigned service tickets and diagnostic assessments
    included in the dispatch plan.

    Your responsibilities are to:

    1. Review the diagnostic notes for every assigned job.
    2. Identify the likely tools required for each job.
    3. Identify specialized tools mentioned or implied by confirmed
       diagnostic requirements.
    4. Identify likely standard-stock spare parts that may be useful.
    5. Check known inventory information before claiming that an item
       is available.
    6. Produce a vehicle loading checklist.
    7. Identify any required item that is unavailable before dispatch.
    8. Identify service jobs that may have a First-Time Fix risk because
       required parts or tools are unavailable.
    9. Confirm that the technician should mark the first appropriate job
       "En Route" when actual travel begins.
    10. Verify the customer notification status if notification-system
        information is available.

    Do not fabricate inventory quantities, part numbers, tools, or
    notification results.
    """,

    expected_output="""
    A Technician Pre-Dispatch Preparation Report containing:

    - Assigned jobs
    - Diagnostic summary for each job
    - Required standard tools
    - Required specialized tools
    - Suggested spare parts
    - Inventory availability
    - Missing inventory items
    - Vehicle loading checklist
    - First-Time Fix risks
    - Dispatch readiness:
        READY
        or
        NOT READY
    - En Route notification verification requirement
    """,

    agent=field_technician_agent,

    context=[
        pre_dispatch_triage_task,
        technician_dispatch_planning_task
    ]
)


# =========================================================
# SOP 02A
# On-Site Diagnosis and Repair-vs-Replacement Decision
# Responsible Agent: FT
# =========================================================

repair_vs_replacement_task = Task(
    description="""
    Perform the Field Technician portion of SOP 02:
    Spare Parts Procurement and Repair-vs-Replacement Decision Matrix.

    On-site diagnostic information:
    {onsite_diagnostic_information}

    Repair cost information:
    {repair_cost}

    Equivalent replacement unit cost:
    {replacement_cost}

    Unit age:
    {unit_age}

    Expected service lifespan:
    {expected_lifespan}

    Repair-vs-Replacement Decision Matrix:
    {repair_replacement_matrix}

    Your responsibilities are to:

    1. Review the actual on-site diagnostic findings.
    2. Identify the confirmed or most-supported faulty component.
    3. Determine whether repair is technically viable.
    4. Apply the Repair-vs-Replacement Decision Matrix.
    5. Calculate or evaluate repair cost relative to replacement cost.
    6. Recommend replacement when:
       - Repair cost exceeds 50% of an equivalent new unit's value, OR
       - The unit is beyond its expected lifespan.
    7. If replacement is recommended, explain the decision clearly.
    8. If repair is economically viable, determine whether the required
       component is available in standard stock.
    9. If a required component is not stocked, prepare a part request for
       the Service Coordinator.

    The part request should include where available:
       - Part/component name
       - Part number
       - Product model
       - Product serial number
       - Quantity
       - Diagnostic reason
       - Urgency
       - Technician notes

    Never invent diagnostic measurements, repair costs, replacement costs,
    expected lifespan, or part numbers.
    """,

    expected_output="""
    A Repair-vs-Replacement Assessment containing:

    - Confirmed diagnostic findings
    - Faulty/suspected component
    - Repair viability
    - Estimated repair cost
    - Equivalent replacement cost
    - Repair cost percentage
    - Unit age
    - Expected lifespan
    - Decision Matrix result
    - Final recommendation:
        REPAIR
        REPLACE
        or
        MORE INFORMATION REQUIRED
    - Reason for recommendation
    - Required parts
    - Standard-stock availability if known
    - Part request details if required
    """,

    agent=field_technician_agent,

    context=[inventory_preparation_task]
)


# =========================================================
# SOP 02B
# Parts Procurement and Customer Delay Management
# Responsible Agent: SC
# =========================================================

parts_procurement_task = Task(
    description="""
    Perform the Service Coordinator portion of SOP 02:
    Spare Parts Procurement and Repair-vs-Replacement Decision Matrix.

    Supplier/parts catalog information:
    {parts_catalog_information}

    Review the Field Technician's repair assessment and part request.

    Only proceed with procurement activities when the Field Technician
    has determined that repair is viable and a non-stocked part is
    required.

    Your responsibilities are to:

    1. Review the requested component.
    2. Verify the part number where sufficient information exists.
    3. Check supplier availability.
    4. Determine supplier lead time.
    5. Determine the estimated part arrival date or ETA.
    6. Record the supplier and availability information.
    7. Prepare the part-order action when appropriate.
    8. Set or recommend the ticket status as "Waiting on Parts".
    9. Prepare a customer communication explaining the parts delay.
    10. Establish a weekly customer-update requirement until the part
        arrives.
    11. Identify the next action once the part becomes available.

    Never fabricate supplier availability, part numbers, prices,
    order confirmations, or ETAs.

    If supplier/catalog data is unavailable, clearly state that the
    availability and ETA must be checked before communicating a confirmed
    delivery estimate.
    """,

    expected_output="""
    A Parts Procurement Record containing:

    - Service ticket reference if available
    - Requested part
    - Part number
    - Quantity
    - Supplier availability
    - Supplier
    - Lead time
    - Estimated arrival date/ETA
    - Procurement/order status
    - Ticket status
    - Customer delay message
    - Weekly update requirement
    - Next customer update
    - Next operational action
    """,

    agent=service_coordinator_agent,

    context=[repair_vs_replacement_task]
)

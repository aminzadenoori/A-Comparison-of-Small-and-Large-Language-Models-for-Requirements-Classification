

# Prompting Strategy and Rationale
We adopt a *Chain-of-Thought (CoT) prompting strategy with examples* for requirement classification. This approach has demonstrated strong theoretical and empirical performance in guiding language models to reason step-by-step, improving interpretability and consistency. By explicitly structuring the classification process into definitions, examples, and reasoning, we reduce ambiguity and promote reproducible labeling decisions.

---

```markdown
### 🧠 COT with Examples Prompt
```text
Let's analyze the classification step by step.

**Step 1: Understand the Definitions:**  
{Definitions}

**Step 2: Review the examples:**  
{Examples}

**Step 3: Apply this understanding to classify the following requirement:**  
Requirement: {Text}

**Step 4: Provide the final label in the format:**  
Label: [Your Class Label Here]
```

---

## Definitions of the Classes
While existing definitions in the literature offer valuable starting points, they often lack the granularity or contextual alignment required for our classification tasks. To address this gap, we systematically extracted relevant definitions from prior works and critically examined their applicability. Where necessary, we refined these definitions to better suit our operational framework, and in cases of significant mismatch, we proposed rewritten versions that preserve the original intent while enhancing clarity and precision. The resulting definitions are presented below.

---

### PROMISE

#### Functional
**Functional Requirement:**  
Functional requirements define the essential functions a system must perform, the services it must offer, and the behaviours it must exhibit under specified conditions. They focus on what the system should do—describing actions, operations, or transformations the system executes—without addressing implementation constraints. They typically specify the inputs (stimuli) to the system, the outputs (responses) from the system, and the behavioural relationships between them.

#### Non-Functional Requirements (NFR)
Non-functional requirements identify any *property*, *characteristic*, *attribute*, *quality*, *constraint*, or *performance aspect* of a system. These requirements are not specifically concerned with the functionality of a system. They place restrictions on the product being developed and the development process, and specify criteria that can be used to judge the operation of a system, rather than specific behaviours.

**Examples include:**
- **Performance**: responsiveness and efficient processing  
- **Scalability**: ability to handle growth  
- **Portability**: cross-platform operation  
- **Compatibility**: interaction with other systems  
- **Reliability**: consistent and failure-free operation  
- **Maintainability**: ease of updates and modifications  
- **Availability**: uptime and accessibility  
- **Security**: protection against unauthorized access  
- **Usability**: user-friendliness  
- **Fault Tolerance**: continued operation under faults  
- **Legal**: compliance with laws and regulations  
- **Look & Feel**: visual and UI consistency  
- **Operational**: system operations 

---

### SecReq

#### Security Requirements
Security requirements are prescriptive constraints imposed on a system’s functional behaviour to operationalize its security goals. They are not functional requirements themselves but restrict how functions are performed to prevent, detect, or recover from harm. Security requirements are derived from business and functional goals and may be categorized as:

- **Primary**: directly supporting fundamental security objectives  
- **Secondary**: enabling primary goals when direct enforcement is infeasible or costly

They address risks, threats, and assets, and influence or are influenced by security mechanisms, vulnerabilities, and attacks.

**Categories include:**
- Confidentiality  
- Integrity  
- Availability  
- Authentication  
- Authorization  
- Non-repudiation  
- Physical Security  
- Regulatory Compliance

#### Non-Security Requirements
Non-security requirements govern system behavior, performance, and structure without being directly tied to security goals. They do not aim to prevent, detect, or recover from harm.

**Examples include:**
- Availability  
- Fault Tolerance  
- Legal  
- Look & Feel  
- Maintainability  
- Operational  
- Performance  
- Portability  
- Scalability  
- Usability  
(Excludes security-specific categories)

---



### PROMISE Reclass

#### Functional Requirements
Functional requirements define the essential functions a system must perform, the services it must offer, and the behaviours it must exhibit under specified conditions. They focus on what the system should do—describing actions, operations, or transformations the system executes—without addressing implementation constraints. They typically specify the inputs (stimuli) to the system, the outputs (responses) from the system, and the behavioural relationships between them.

#### Non-Functional Requirements
Non-functional (negation of functional): Non-functional requirements do not define the essential functions a system must perform, the services it must offer, or the behaviours it must exhibit under specified conditions. They do not focus on what the system should do—avoiding descriptions of actions, operations, or transformations the system executes—and instead address implementation constraints. They typically exclude specifications of inputs (stimuli) to the system, outputs (responses) from the system, and behavioural relationships between them.

#### Quality Requirements
A quality requirement expresses how well a system or service should execute an intended function. They include attributes or constraints that address product quality aspects and quality in use aspects.

**Product quality aspects include:**
- Functional Suitability  
- Reliability  
- Performance Efficiency  
- Usability  
- Maintainability  
- Security  
- Compatibility  
- Portability  


#### Non-Quality Requirements
A non-quality requirement does not express how well a system or service should execute an intended function. They exclude attributes or constraints that address product quality aspects and quality in use aspects. Non-quality requirements focus on specific functions rather than global system properties.
```

---

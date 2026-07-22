# Nedbank Africa Expansion — Public Sentiment Prediction

> **MiroFish Simulation Requirement Preset**
> Version: v1.0.0 | Created: 2026-07-10

---

## Project Name

Nedbank Africa Expansion Sentiment Prediction

## Simulation Requirement

Nedbank, one of South Africa's largest retail banks, announces a strategic
expansion into the rest of the African continent, opening branches and digital
banking services in Nigeria, Kenya, Ghana, and Mozambique over the next 18
months. The expansion includes partnerships with local telecom providers for
mobile money integration, a US$200 million investment fund, and a marketing
campaign positioning Nedbank as "the bank that understands Africa." Predict the
public sentiment response across social media platforms (Twitter and Reddit) in
the days following the announcement. Consider the perspectives of South African
customers reacting to capital outflow concerns, Nigerian and Kenyan citizens'
reception of a South African bank entering their markets, fintech competitors
(like MTN MoMo, Flutterwave, OPay), financial analysts discussing the investment
rationale, regulators in each target country, pan-African economic nationalists,
and ordinary citizens concerned about data privacy and neo-colonial financial
dominance. How does sentiment evolve over a 72-hour period across these diverse
stakeholder groups?

## Additional Context

Key factors to model:

1. South-South investment dynamics and historical skepticism toward South African corporate expansion in Africa
2. Competitive landscape with established fintech players (Flutterwave, OPay, Paystack, MPesa)
3. Mobile money integration as a differentiator
4. Regulatory uncertainty in each target market
5. Currency volatility and capital controls concerns from SA depositors
6. Nedbank's ESG commitments and "Africa-focused" brand positioning
7. Data sovereignty concerns in target countries
8. Comparison to prior pan-African bank expansions (Standard Bank, Absa/Barclays)

The simulation should capture the initial announcement hype, analyst
commentary, competitive response, regulatory statements, and citizen reactions
across geographic and demographic segments.

## Recommended Entity Types

Suggested agent personas for the knowledge graph:

| Entity Type | Description |
|---|---|
| Customer | General retail banking customer |
| BankingCustomer | Nedbank-specific SA customer |
| FintechCompetitor | Competitor fintech (Flutterwave, OPay, Paystack, MPesa, MTN MoMo) |
| FinancialAnalyst | Market analyst covering African banking |
| Regulator | Financial regulator in target country |
| Journalist | Business/tech journalist covering the story |
| EconomicNationalist | Pan-African economic sovereignty advocate |
| Citizen | Ordinary citizen in a target market |
| TelecomPartner | Local telecom partner providing mobile money rails |
| GovernmentOfficial | Government official in target country |
| Investor | Nedbank shareholder / institutional investor |
| SocialMediaInfluencer | Influencer shaping public discourse |

## Recommended Platforms

| Platform | Enabled |
|---|---|
| Twitter | Yes |
| Reddit | Yes |

## Recommended Time Configuration

| Parameter | Value |
|---|---|
| Total simulation hours | 72 |
| Minutes per round | 30 |
| Approx. total rounds | 144 |

## Suggested Source Documents

Upload these (or similar) documents when creating the MiroFish project so the
ontology generator and knowledge graph have rich source material:

1. Nedbank annual report and Africa strategy disclosure (PDF)
2. Press release on the expansion announcement (MD)
3. Market analysis reports on African banking sector (PDF/MD)
4. Articles on Standard Bank and Absa prior pan-African expansion outcomes (MD)
5. Fintech landscape reports for Nigeria, Kenya, Ghana, Mozambique (MD)
6. Social media sentiment studies on South African brands entering African markets (TXT)

## How to Use This Preset

1. **Create project** — In the MiroFish frontend, enter the **Project Name**
   above, paste the **Simulation Requirement** and **Additional Context** into
   the project creation form.
2. **Upload documents** — Upload the suggested source documents (or equivalent
   real-world materials).
3. **Generate ontology** — The pipeline auto-generates entity/edge types from
   the documents and requirement.
4. **Build graph** — The knowledge graph is constructed from the ontology.
5. **Generate profiles** — Agent personas are created for each entity in the
   graph.
6. **Run simulation** — Twitter + Reddit simulation runs for 72 simulated hours.
7. **Generate report** — The Report Agent produces a sentiment prediction report
   across all stakeholder groups.

## Usage via API

```bash
# Create project with ontology generation (Endpoint 1)
curl -X POST http://localhost:5001/api/graph/ontology/generate \
  -F "project_name=Nedbank Africa Expansion Sentiment Prediction" \
  -F "simulation_requirement=Nedbank, one of South Africa's largest retail banks, announces a strategic expansion into the rest of the African continent, opening branches and digital banking services in Nigeria, Kenya, Ghana, and Mozambique over the next 18 months. The expansion includes partnerships with local telecom providers for mobile money integration, a US\$200 million investment fund, and a marketing campaign positioning Nedbank as 'the bank that understands Africa.' Predict the public sentiment response across social media platforms (Twitter and Reddit) in the days following the announcement. Consider the perspectives of South African customers reacting to capital outflow concerns, Nigerian and Kenyan citizens' reception of a South African bank entering their markets, fintech competitors (like MTN MoMo, Flutterwave, OPay), financial analysts discussing the investment rationale, regulators in each target country, pan-African economic nationalists, and ordinary citizens concerned about data privacy and neo-colonial financial dominance. How does sentiment evolve over a 72-hour period across these diverse stakeholder groups?" \
  -F "additional_context=Key factors to model: (1) South-South investment dynamics and historical skepticism toward South African corporate expansion in Africa; (2) Competitive landscape with established fintech players (Flutterwave, OPay, Paystack, MPesa); (3) Mobile money integration as a differentiator; (4) Regulatory uncertainty in each target market; (5) Currency volatility and capital controls concerns from SA depositors; (6) Nedbank's ESG commitments and 'Africa-focused' brand positioning; (7) Data sovereignty concerns in target countries; (8) Comparison to prior pan-African bank expansions (Standard Bank, Absa/Barclays). The simulation should capture the initial announcement hype, analyst commentary, competitive response, regulatory statements, and citizen reactions across geographic and demographic segments." \
  -F "files=@nedbank_annual_report.pdf" \
  -F "files=@africa_banking_market_analysis.md" \
  -F "files=@fintech_landscape_west_africa.md"

# The response contains project_id — use it for subsequent steps:
#   2. Build graph:    POST /api/graph/build
#   3. Create sim:     POST /api/simulation/create
#   4. Prepare sim:    POST /api/simulation/prepare
#   5. Start sim:      POST /api/simulation/start
#   6. Generate report: POST /api/report/generate
```
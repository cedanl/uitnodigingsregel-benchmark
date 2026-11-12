# uitnodigingsregel-benchmark

Meet hoe goed modellen uitval voorspellen op de data van je eigen instelling, en welke variabelen daaraan bijdragen. De benchmark draait op je eigen computer; er gaat geen studentdata naar buiten. De uitkomst kun je delen. De gedeelde uitkomsten vormen samen een [gezamenlijk overzicht](https://cedanl.github.io/uitnodigingsregel-benchmark/) van welke variabelen ertoe doen en hoe goed de modellen voorspellen.

## Installeren

Je hebt Python 3.13 of hoger en [uv](https://astral.sh/uv/) nodig.

```bash
git clone https://github.com/cedanl/uitnodigingsregel-benchmark.git
cd uitnodigingsregel-benchmark
uv sync
```

Op een Mac heeft het model XGBoost het onderdeel OpenMP nodig. Dat installeert `uv sync` niet. Zonder OpenMP slaat de benchmark XGBoost over en draait de andere modellen, waaronder Hist Gradient Boosting: hetzelfde soort model, uit scikit-learn. Wil je XGBoost erbij:

```bash
brew install libomp
```

## Welke data heb je nodig

Eén CSV-bestand met studenten van wie de afloop bekend is. Dat is hetzelfde bestand waarmee je de Uitnodigingsregel traint. Een bestand met het huidige cohort is niet nodig: daarvan is de afloop nog niet bekend, dus daar kun je niet op toetsen. De benchmark verdeelt het bestand zelf in een deel om te trainen en een deel om te toetsen.

- **Eén rij per student**, met eerstejaars die in hetzelfde jaar zijn begonnen of in een paar opeenvolgende jaren.
- **Een kolom met de afloop**: 1 als de student is uitgevallen, 0 als niet. Standaard heet die kolom `Dropout`.
- **Kenmerken van de student**, bij voorkeur met de namen uit de [gedeelde variabelenlijst](#gedeelde-variabelen), zoals leeftijd, aantal aanmeldingen, vooropleiding en verzuim. Eigen kolommen mogen erbij.
- **Optioneel** een kolom met het studentnummer (wordt niet gebruikt om te voorspellen) en een kolom met het cohort (alleen nodig om op een later jaar te toetsen).

Twee dingen bepalen of de uitkomst iets zegt:

1. **Genoeg uitvallers.** Reken op minstens 150 uitvallers in het bestand, zodat de testgroep er rond de 30 heeft. Bij 15% uitval zijn dat ongeveer 1000 studenten. Heb je er minder, voeg dan meerdere jaren samen. Bij minder dan 30 uitvallers in de testgroep geeft de benchmark een waarschuwing.
2. **Alleen gegevens die bekend zijn op het moment dat je uitnodigt.** Nodig je in week 10 uit, dan hoort verzuim uit week 11 of later er niet in. Anders lijkt een model beter dan het in de praktijk is. De benchmark kan dat niet zelf controleren.

Gebruik dezelfde doelgroep als bij de Uitnodigingsregel, eerstejaars BOL. Met andere groepen erbij zijn de uitkomsten niet te vergelijken met die van andere instellingen.

## Gebruiken

Start de app:

```bash
uv run uitnodigingsregel-benchmark
```

De app opent in je browser. Stoppen doe je met Ctrl+C in het terminalvenster.

1. **Kies je data.** Upload een CSV met één rij per student, een kolom die aangeeft of de student is uitgevallen (1) of niet (0), en de kenmerken van de student. Een kolom met studentnummers is niet nodig. Of klik op de demo-data om de app eerst te proberen.
2. **Controleer de kolommen.** De app herkent de uitvalkolom, het studentnummer en een cohortkolom, stelt voor welke kolommen samen één variabele vormen, en koppelt je kolommen aan de gedeelde variabelenlijst. Pas aan wat niet klopt. Kan de data niet worden gebruikt, dan staat er direct waarom.
3. **Draai de benchmark.** Kies snel (een eerste controle, ongeveer een minuut) of volledig (alle modellen, vijf keer met een andere splitsing van de data, voor een uitkomst om te delen), of de impact per variabele wordt gemeten, en eventueel een variabele om mee te vergelijken (bijvoorbeeld verzuim).
4. **Lees de uitkomst.** Per model de precisie en recall als de 20% studenten met de hoogste voorspelde kans worden uitgenodigd, en een grafiek met de impact per variabele.
5. **Download het rapport.** Eén HTML-bestand met de uitkomst en de grafieken, om te bewaren of door te sturen naar collega's. Het opent in elke browser, ook zonder internet.
6. **Deel de uitkomst.** Download het resultaatbestand en lever het aan voor het gezamenlijke overzicht (zie hieronder). Het bestand bevat de uitkomsten per model en aantallen studenten. Het bevat geen namen van je eigen kolommen en geen gegevens van individuele studenten (zie [Gedeelde variabelen](#gedeelde-variabelen)).
7. **Vergelijk.** De app toont je uitkomst naast de gedeelde uitkomsten van andere instellingen. De app haalt die alleen op; er wordt niets over je data verstuurd.

Het rapport en het resultaatbestand staan ook in de map `results/`.

## De uitkomst lezen

Alle getallen gaan over hetzelfde scenario: je nodigt de 20% studenten uit die volgens het model de grootste kans op uitval hebben.

| Getal | Wat het is | Wat het voor jou betekent |
|---|---|---|
| Precisie | Het deel van de uitgenodigde studenten dat ook echt uitviel | Hoe vaak een gesprek terecht is. Bij 40% is vier van de tien uitgenodigde studenten later uitgevallen |
| Recall | Het deel van alle uitvallers dat bij de uitgenodigden zat | Hoeveel uitvallers je bereikt. Bij 35% zit ruim een derde van alle uitvallers in je uitnodigingen |
| Uitval in de data | Het deel van de studenten in de testgroep dat uitviel | De ondergrens: dit is de precisie als je willekeurig uitnodigt |
| Oordeel (stoplicht) | Groen, oranje of rood, op basis van precisie en recall | Groen: betrouwbaar. Oranje: gebruik met voorzichtigheid. Rood: niet bruikbaar, het model voegt te weinig toe |
| Lift | Precisie gedeeld door het uitvalpercentage | Hoeveel keer beter dan willekeurig uitnodigen. Een lift van 2 betekent twee keer zoveel terechte gesprekken |
| ± | De spreiding over vijf splitsingen van de data | Hoeveel het getal afhangt van welke studenten toevallig in de testgroep zaten. Een verschil kleiner dan de spreiding is toeval |
| Willekeurig uitnodigen | Referentie zonder model | Wat je haalt zonder enige voorspelling |
| Uitnodigen op één variabele | Referentie zonder model: de 20% met de hoogste waarde op één variabele, bijvoorbeeld verzuim | Wat je haalt met een eenvoudige regel. Haalt een model niet meer, dan levert het weinig op |
| Impact van een variabele | Het verschil in precisie tussen het model met en zonder die variabele, in procentpunten | Positief: de variabele helpt. Rond nul: de variabele voegt in jouw data niets toe. Een grijze balk valt binnen de spreiding |

**Willekeurige splitsing of later cohort.** Bij een willekeurige splitsing zitten studenten uit dezelfde jaren in de trainings- en de testgroep. Dat valt vaak gunstiger uit dan de praktijk, waarin je een nieuw jaar voorspelt. Toetsen op een later cohort lijkt meer op de praktijk. Vergelijk daarom alleen uitkomsten met dezelfde soort splitsing; het gezamenlijke overzicht doet dat ook.

**De impact lezen.** Kijk eerst naar de kleur: alleen groene en oranje balken zijn duidelijker dan toeval. Een variabele zonder impact is niet per se onbelangrijk: als twee variabelen hetzelfde zeggen, zoals verzuim in week 3 en verzuim in week 4, neemt de ene het over als de andere wegvalt. Beide lijken dan weinig toe te voegen. In `results/voorbeeld/` staat een voorbeeld, gemaakt op de demo-data.

## Je uitkomst delen

Gedeelde uitkomsten staan in het [gezamenlijke overzicht](https://cedanl.github.io/uitnodigingsregel-benchmark/). Het overzicht is openbaar: je uitkomst staat er onder de naam die je in de app hebt ingevuld. Gebruik een code als naam als je niet herkenbaar wilt zijn.

1. Draai de benchmark met "Volledig". Een snelle run telt niet mee.
2. Download het resultaatbestand in de app.
3. Mail het bestand naar je contactpersoon bij CEDA. De app verstuurt zelf niets.
4. CEDA controleert het bestand en voegt het toe aan deze repository. Het overzicht wordt daarna vanzelf opnieuw gebouwd.

Een uitkomst staat dus pas in het overzicht nadat CEDA het bestand heeft toegevoegd. Bestanden in je eigen map `results/` staan alleen op je eigen computer.

### Voor CEDA: een gedeelde uitkomst opnemen

1. Controleer of de naam of code in het bestand (`instelling_id`) er een is waar de instelling mee akkoord is, en of het een volledige run is (`quick_mode` is false).
2. Voeg het bestand in een pull request toe aan `results/`. De CI controleert het bestand: het formaat, de versie en of er alleen namen uit de gedeelde variabelenlijst in staan. Een afgekeurd bestand maakt de pull request rood; vraag de instelling dan om de benchmark opnieuw te draaien met de laatste versie. Lokaal controleren: `uv run python -m uitnodigingsregel_benchmark.check results/<bestand>.json`.
3. Merge de pull request.
4. De workflow `Overview` bouwt de pagina opnieuw en publiceert die op GitHub Pages. Uitkomsten van snelle runs en van de demo-data blijven buiten het overzicht.

Het overzicht lokaal bouwen: `uv run python -m uitnodigingsregel_benchmark.overview`, en open daarna `site/index.html`. Lokaal tellen ook de bestanden mee die alleen in je eigen map `results/` staan.

## Gedeelde variabelen

Elke instelling noemt haar kolommen anders. Om uitkomsten bij elkaar op te tellen, gebruikt het resultaatbestand alleen de namen uit een gedeelde lijst: `src/uitnodigingsregel_benchmark/metadata/variabelen.yaml`. De lijst volgt de variabelendefinities van de Uitnodigingsregel, met per variabele een omschrijving.

- Een kolom met dezelfde naam als een gedeelde variabele wordt vanzelf gekoppeld. Hoofdletters, spaties en lage streepjes maken niet uit.
- Heet een kolom anders maar bedoelt hij hetzelfde, koppel hem dan in de app bij stap 2. De app bewaart die keuze in `koppeling.json` op je eigen computer; dat bestand wordt niet gedeeld.
- Een kolom die niet in de lijst staat telt gewoon mee, maar heet in het resultaatbestand `eigen_variabele_1`, `eigen_variabele_2`, enzovoort. Een eigen groep van kolommen heet `eigen_groep_1`. Je rapport en de app tonen welke kolom erbij hoort; het resultaatbestand niet.
- Gedeelde variabelen die samen één kenmerk vormen, zoals de kolommen per niveau van de vooropleiding, tellen bij de impactmeting altijd als één variabele.

Eigen variabelen tellen niet mee in het gezamenlijke overzicht: `eigen_variabele_1` is bij elke instelling iets anders.

### Een variabele voorstellen voor de gedeelde lijst

Meet je een variabele die voor andere instellingen ook bruikbaar is, zoals reistijd of een toetsuitslag? Maak dan een issue aan in deze repository met de naam, een omschrijving van één zin, hoe je de waarde berekent en uit welk systeem die komt. Na opname in de lijst koppelt de app de kolom vanzelf.

## Commandoregel en instellingenbestand

Voor gevorderde gebruikers. Alles wat de app doet, kan ook vanaf de commandoregel, met de instellingen in `config.yaml`.

Probeer de benchmark eerst op de demo-data:

```bash
uv run uitnodigingsregel-benchmark --demo --quick
```

Draai hem daarna op je eigen data:

1. Zet je CSV in `data/01-raw/user_data/`: één bestand met een kolom `Dropout` (1 = uitgevallen, 0 = niet) en de kenmerken van de student. Een kolom met studentnummers is niet nodig.
2. Vul `instelling_id` en `dropout_year` in `config.yaml` in.
3. Start:

```bash
uv run uitnodigingsregel-benchmark --train data/01-raw/user_data/data.csv
```

Uitvoer: `results/benchmark_{instelling_id}_{dropout_year}.json`, en het rapport als HTML-bestand met dezelfde naam.

Vóór het trainen controleert de benchmark de data. Ontbreekt er iets of klopt er iets niet, dan stopt hij en staat er wat je moet aanpassen.

| Optie | Wat het doet |
|---|---|
| (geen) | Start de app |
| `--train PAD` | Draai op je eigen CSV |
| `--demo` | Draai op de demo-data; schrijft `results/benchmark_demo_*.json` |
| `--quick` | Drie modellen, minder zoekrondes en één splitsing, voor een eerste controle. In het resultaatbestand staat `quick_mode: true`; deel een snelle uitkomst niet |
| `--cohort-column NAAM` | Toets op een later cohort (zie hieronder) |
| `--impact` | Meet de impact per variabele (zie hieronder) |
| `--config PAD` | Gebruik een ander instellingenbestand dan `config.yaml` |
| `--results-dir PAD` | Schrijf de uitkomst naar een andere map |
| `--no-browser` | Start de app zonder de browser te openen |

## Instellingen

Alle instellingen staan in `config.yaml`, met uitleg per regel. Er hoeven geen Python-bestanden te worden aangepast.

| Instelling | Betekenis | Standaard |
|---|---|---|
| `instelling_id` | Korte code voor je instelling, komt in de bestandsnaam | `anoniem` |
| `dropout_year`, `training_years` | Jaren waarover de data gaat | geen |
| `dropout_column` | Naam van de uitvalkolom | `Dropout` |
| `id_column` | Naam van de kolom met studentnummers; mag ontbreken | `Studentnummer` |
| `cohort_column` | Naam van de cohortkolom (zie hieronder) | geen |
| `models` | Modellen om te draaien | alle |
| `n_iter`, `cv_folds` | Zoekrondes per model en aantal delen per ronde; lager is sneller | 20, 5 |
| `test_size` | Aandeel van de data om op te toetsen bij een willekeurige splitsing | 0.2 |
| `repeats` | Aantal keren dat de willekeurige splitsing wordt herhaald (zie hieronder) | 5 (snel: 1) |
| `n_jobs` | Aantal processen tegelijk; zet op 2 als het geheugen volraakt | alle kernen |
| `impact`, `impact_variables`, `variable_groups` | Impact per variabele (zie hieronder) | uit |
| `reference_variable` | Kolom voor de referentie "uitnodigen op één variabele", bijvoorbeeld `absence_unauthorized` | geen |
| `column_mapping` | Eigen kolom gekoppeld aan een gedeelde variabele, bijvoorbeeld `{Leeftijd: StudentAge}`. Zonder deze instelling leest de commandoregel `koppeling.json` uit de app | geen |

## Toetsen op een later cohort

Standaard wordt de data willekeurig gesplitst: 80% om de modellen te trainen en 20% om ze te toetsen. In de praktijk traint een instelling op eerdere cohorten en voorspelt het volgende. Een willekeurige splitsing geeft vaak een gunstigere uitkomst dan dat.

Heeft je data een kolom met het cohort (bijvoorbeeld het startjaar) en minstens twee cohorten, dan kan de benchmark trainen op de eerdere cohorten en toetsen op het laatste. In de app kies je de kolom bij stap 2. Vanaf de commandoregel zet je de naam in `config.yaml`:

```yaml
cohort_column: "Startjaar"
```

of geef je hem mee met `--cohort-column Startjaar`. De cohortkolom wordt niet gebruikt om te voorspellen. `dropout_year` en `training_years` in de uitkomst volgen dan uit de data.

Dit is optioneel. Zonder cohortkolom, of met maar één cohort in de data, gebruikt de benchmark de willekeurige splitsing. In het resultaatbestand staat bij `metadata.split` welke splitsing is gebruikt, en het gezamenlijke overzicht vergelijkt alleen uitkomsten met dezelfde soort splitsing.

## De impact van een variabele meten

Met de impactmeting traint de benchmark elk model opnieuw zonder één variabele, en slaat het verschil op in precisie en recall bij 20% uitgenodigde studenten. Dat laat zien wat een variabele oplevert; een rangschikking van variabelen doet dat niet. In de app is het een vinkje bij stap 3. Vanaf de commandoregel:

```bash
uv run uitnodigingsregel-benchmark --train data/01-raw/user_data/data.csv --impact
```

In de uitkomst staat per model een lijst `variable_impact` met voor elke variabele:

| Veld | Betekenis |
|---|---|
| `precision_without`, `recall_without` | Precisie en recall bij 20% (in %) van het model dat zonder de variabele is getraind |
| `precision_diff`, `recall_diff` | Uitkomst met de variabele min de uitkomst zonder, in procentpunten. Positief: het model voorspelt beter met de variabele. Rond nul of negatief: de variabele voegt in deze data niets toe |

De run duurt langer: per variabele wordt elk model één keer extra getraind. De modellen houden de instellingen die bij het zoeken zijn gevonden. Wil je alleen bepaalde variabelen toetsen, noem ze dan in `config.yaml`:

```yaml
impact_variables: [StudentAge, Vooropleiding]
```

Een variabele die over meerdere kolommen is verdeeld, zoals een kolom per niveau van de vooropleiding, telt als één variabele als je de kolommen samenvoegt in een groep. De app stelt zulke groepen zelf voor. In `config.yaml` staat een `*` voor een willekeurig stuk tekst:

```yaml
variable_groups:
  Vooropleiding: ["VooroplNiveau_*"]
  Richting: [Economie, Landbouw, Techniek]
```

Twee variabelen die dezelfde informatie dragen kunnen elk onbelangrijk lijken, omdat de ander het overneemt.

## Spreiding over meerdere splitsingen

Bij één willekeurige splitsing hangt de uitkomst af van welke studenten toevallig in de testgroep zitten. Bij een testgroep van 200 studenten kan de precisie daardoor enkele procentpunten verschuiven. Een volledige run herhaalt daarom de splitsing vijf keer (`repeats`), telkens met andere studenten in de testgroep. De uitkomst is het gemiddelde; de spreiding staat erachter als ±.

- In het resultaatbestand staan `precision_sd` en `recall_sd` per model, en `precision_diff_sd` en `recall_diff_sd` per variabele. Het aantal herhalingen staat in `metadata.split.repeats`.
- Valt een verschil binnen de spreiding, dan toont de app en het rapport dat: een grijze balk bij de impact, en bij de modellen een melding als het verschil met het beste model binnen de spreiding valt.
- Het zoeken naar de beste instellingen per model gebeurt één keer, op de eerste splitsing. Bij de andere splitsingen worden de modellen met die instellingen opnieuw getraind. Dat scheelt veel tijd.
- Bij toetsen op een later cohort is er maar één mogelijke testgroep; die wordt niet herhaald.

## Uitkomsten van verschillende versies vergelijken

Sinds `student_signal` 0.2.0 wordt het studentnummer (`Studentnummer`) niet meer gebruikt om te voorspellen. Uitkomsten van een eerdere versie zijn niet te vergelijken met uitkomsten van 0.2.0 of later.

In het resultaatbestand staan drie versies: van het formaat (`metadata.format_version`), van de benchmark (`metadata.benchmark_version`) en van `student_signal` (`metadata.student_signal_version`). Het formaat ligt vast in `src/uitnodigingsregel_benchmark/metadata/result_schema.json`. Vóór het schrijven controleert de benchmark het bestand daartegen; een bestand dat niet voldoet wordt niet geschreven. Het gezamenlijke overzicht slaat bestanden met een onbekende versie over en meldt welke. Een bestand zonder `format_version` is met een eerdere versie gemaakt: draai de benchmark opnieuw.

Heeft de testgroep minder dan 30 uitvallers, dan meldt de benchmark dat: de precisie hangt dan af van een handvol studenten.

## Modellen toevoegen

De beschikbare modellen staan in `src/uitnodigingsregel_benchmark/models.py`. Elk model dat met scikit-learn samenwerkt kan daar worden toegevoegd:

```python
MODELS = {
    ...
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "use_scaled": False,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
        },
    },
}
```

Uitkomsten met andere modellen dan de standaardmodellen zijn voor eigen gebruik; het gezamenlijke overzicht vergelijkt de standaardmodellen.

## Ontwikkelen

```bash
uv run pytest
```

## Licentie

MIT

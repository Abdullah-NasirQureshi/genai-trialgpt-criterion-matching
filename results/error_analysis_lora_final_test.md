# Error analysis (lora_final, test)

58 of 206 criterion predictions have a wrong label. Counts per category: A false_certainty: 27, B missed_info: 21, D unsupported: 29, E wrong_evidence: 7, F other: 5

## A false_certainty  (27 cases)

### Case 1: annotation 174 (patient sigir-201410, NCT02097186, exclusion)
- **Criterion:** Patients with previous axillary surgery
- **Gold:** not enough information | evidence []: (none)
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning. [1] She is now complaining of a cool right foot. [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered. [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 2: annotation 193 (patient sigir-201411, NCT01526382, exclusion)
- **Criterion:** Patients were moribund.
- **Gold:** not enough information | evidence [0, 1, 2, 3, 4]: [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.; [1] She denies trauma.; [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.; [3] Her body temperature is normal and her blood pressure is 80/60.; [4] Her right arm has no discoloration or movement limitation.
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. [1] She denies trauma. [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. [3] Her body temperature is normal and her blood pressure is 80/60. [4] Her right arm has no discoloration or movement limitation. [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 3: annotation 325 (patient sigir-201417, NCT02556359, inclusion)
- **Criterion:** Genetic patients
- **Gold:** not enough information | evidence []: (none)
- **Predicted:** not included | evidence [0]: [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
- **Note:** [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40. [1] The physical examination is remarkable for tenderness and positive Murphy sign. [2] Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid. [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained. [4] After surgery, the patient is taken to the ICU. [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## B missed_info  (21 cases)

### Case 4: annotation 162 (patient sigir-201410, NCT02097186, inclusion)
- **Criterion:** Patients undergoing open abdominal aortic aneurysm repair
- **Gold:** not included | evidence []: (none)
- **Predicted:** not enough information | evidence [0]: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- **Note:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning. [1] She is now complaining of a cool right foot. [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered. [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 5: annotation 185 (patient sigir-201411, NCT01526382, inclusion)
- **Criterion:** Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- **Gold:** included | evidence [2, 3]: [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.; [3] Her body temperature is normal and her blood pressure is 80/60.
- **Predicted:** not enough information | evidence []: (none)
- **Note:** [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. [1] She denies trauma. [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. [3] Her body temperature is normal and her blood pressure is 80/60. [4] Her right arm has no discoloration or movement limitation. [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 6: annotation 188 (patient sigir-201411, NCT01526382, inclusion)
- **Criterion:** The use of vasopressors to maintain a systolic blood pressure of at least 90mmHg despite fluid resuscitation, low dose of dopamine (≤ 5 μg/kg per minute), or dobutamine;
- **Gold:** not included | evidence [3]: [3] Her body temperature is normal and her blood pressure is 80/60.
- **Predicted:** not enough information | evidence []: (none)
- **Note:** [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. [1] She denies trauma. [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. [3] Her body temperature is normal and her blood pressure is 80/60. [4] Her right arm has no discoloration or movement limitation. [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## D unsupported  (29 cases)

### Case 7: annotation 389 (patient sigir-201421, NCT00997100, exclusion)
- **Criterion:** Severe renal impairment (estimated or measured GFR <50%)
- **Gold:** not enough information | evidence []: (none)
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 21-year-old female is evaluated for progressive arthralgias and malaise. [1] On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles. [2] Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA. [3] Her urine is positive for protein and RBC casts. [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 8: annotation 418 (patient sigir-201422, NCT00723788, exclusion)
- **Criterion:** Failure to pass MRI metal screening
- **Gold:** not enough information | evidence []: (none)
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 15-year-old girl presents to the ER with abdominal pain. [1] The pain appeared gradually and was periumbilical at first, localizing to the right lower quadrant over hours. [2] She has had no appetite since yesterday but denies diarrhea. [3] She has had no sexual partners and her menses are regular. [4] On examination, she has localized rebound tenderness over the right lower quadrant. [5] On an abdominal ultrasound, a markedly edematous appendix is seen. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 9: annotation 683 (patient sigir-20158, NCT02562040, exclusion)
- **Criterion:** Previous tonsillectomy, including partial tonsillectomy
- **Gold:** not applicable | evidence []: (none)
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings. [1] No history of headache or night terrors. [2] The boy's teacher recently contacted his parents because she was concerned about his declining grades, lack of attention, and excessive sleepiness during class. [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## E wrong_evidence  (7 cases)

### Case 10: annotation 179 (patient sigir-201410, NCT02264964, exclusion)
- **Criterion:** Had been performed central venous puncture or catheterization before.
- **Gold:** not excluded | evidence [0]: [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning. [1] She is now complaining of a cool right foot. [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered. [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 11: annotation 186 (patient sigir-201411, NCT01526382, inclusion)
- **Criterion:** Heart rate of at least 90/min;
- **Gold:** included | evidence [2]: [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- **Predicted:** included | evidence [3]: [3] Her body temperature is normal and her blood pressure is 80/60.
- **Note:** [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. [1] She denies trauma. [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. [3] Her body temperature is normal and her blood pressure is 80/60. [4] Her right arm has no discoloration or movement limitation. [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 12: annotation 197 (patient sigir-201411, NCT01723137, exclusion)
- **Criterion:** Decreased level of consciousness
- **Gold:** not excluded | evidence [0, 5]: [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.; [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. [1] She denies trauma. [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. [3] Her body temperature is normal and her blood pressure is 80/60. [4] Her right arm has no discoloration or movement limitation. [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## F other  (5 cases)

### Case 13: annotation 422 (patient sigir-201422, NCT00236912, inclusion)
- **Criterion:** If female, using birth control
- **Gold:** not applicable | evidence [3]: [3] She has had no sexual partners and her menses are regular.
- **Predicted:** not enough information | evidence [3]: [3] She has had no sexual partners and her menses are regular.
- **Note:** [0] A 15-year-old girl presents to the ER with abdominal pain. [1] The pain appeared gradually and was periumbilical at first, localizing to the right lower quadrant over hours. [2] She has had no appetite since yesterday but denies diarrhea. [3] She has had no sexual partners and her menses are regular. [4] On examination, she has localized rebound tenderness over the right lower quadrant. [5] On an abdominal ultrasound, a markedly edematous appendix is seen. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 14: annotation 743 (patient sigir-201513, NCT01255670, inclusion)
- **Criterion:** patient has daily access to his/her e-mail
- **Gold:** not enough information | evidence [4]: [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Predicted:** not applicable | evidence []: (none)
- **Note:** [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes. [1] He is toxic-appearing, and leans forward while sitting on his mother's lap. [2] He is drooling and speaks with a muffled "hot potato" voice. [3] The parents deny the possibility of foreign body ingestion or trauma, and they report that they are delaying some of his vaccines. [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 15: annotation 745 (patient sigir-201513, NCT01255670, inclusion)
- **Criterion:** female patients have adequate birth-control method
- **Gold:** not applicable | evidence []: (none)
- **Predicted:** not enough information | evidence []: (none)
- **Note:** [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes. [1] He is toxic-appearing, and leans forward while sitting on his mother's lap. [2] He is drooling and speaks with a muffled "hot potato" voice. [3] The parents deny the possibility of foreign body ingestion or trauma, and they report that they are delaying some of his vaccines. [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

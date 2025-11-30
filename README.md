
## 1. Wprowadzenie

#### 1.1 Cel projektu

Celem projektu jest zbudowanie interaktywnego narzędzia analityczno - decyzyjnego, które wykorzystuje dostarczone dane do wizualizacji, porównywania i optymalizacji śladu węglowego sprzętu i infrastruktury IT. Narzędzie ma być zdolne do obliczania ekologicznych oszczędności, wynikających z przedłużenia życia urządzenia.

Rozwiązanie ma również pomóc firmom w optymalnym zarządzaniu cyklem życia sprzętu, w tym w podejmowaniu decyzji o przedłużaniu użytkowania oraz w wyborze produktów zaprojektowanych z myślą o łatwiejszym recyklingu (mniej skomplikowane komponenty).

Poprzez integrację danych środowiskowych ($gwp\_total$, $yearly\_tec$) z danymi operacyjnymi system wspiera decyzje mające na celu:

-   **Redukcję Śladu Węglowego (Emisji GHG):** Poprzez wybór niskoemisyjnego sprzętu przy zakupach.

-   **Minimalizację Elektrośmieci:** Poprzez wydłużanie cyklu życia sprzętu i świadome zarządzanie jego wycofaniem z użytku.

#### 1.2 Ekologiczny problem

Sektor IT jest głównym źródłem elektrośmieci. Sprzęt IT zawiera cenne, ale również toksyczne materiały takie jak: metale ciężkie bądź rzadkie pierwiastki. Skracanie cyklu życia sprzętu, napędzane szybkimi zmianami technologicznymi i niewystarczającym planowaniem, prowadzi do masowego, przedwczesnego wyrzucania sprzętu. Dane takie jak przewidywana żywotność są często ignorowane w planowaniu infrastruktury, dlatego stworzona aplikacja będzie zwracała uwagę na ten ważny aspekt.

#### 1.4 Docelowi klienci

Aplikacja CycleFlow jest skierowana do organizacji, które posiadają znaczną infrastrukturę IT (centra danych, biura, zdalni pracownicy) i są zdeterminowane do poprawy swojej efektywności operacyjnej, kosztowej oraz spełnienia celów zrównoważonego rozwoju (ESG).

Wyróżniamy w tym dwie główne grupy docelowe:

1.  Rosnące firmy technologiczne i średnie przedsiębiorstwa

    Są to firmy, które dynamicznie się rozwijają, zwiększają zatrudnienie i w szybkim tempie nabywają nowy sprzęt IT (laptopy, stacje robocze) oraz rozbudowują swoją infrastrukturę chmurową/serwerową.

2.  Duże firmy i liderzy rynku

    W tą grupę wliczją się globalne korporacje, banki, operatorzy telekomunikacyjni i firmy o ugruntowanej pozycji rynkowej, często posiadające również centra danych.

## 2. Architektura Systemu

Warstwa wizualna została wykonana za pomocą biblioteki Streamlit. Backend aplikacji stanowi API wykorzystujące Ollama do wywoływania modeli.

W pierwszej części użytkownik precyzuje swoje potrzeby w celu wybrania optymalnej specyfikacji urządzenia do jego potrzeb. System rekomenduje odpowiednie propozycje urządzeń zwracając szczególną uwagę na ślad węglowy danego urządzenia. Istnieją 3 główne kryteria oceny: wpływ na środowisko, wydajność oraz cena. Model językowy wyjaśnia korzyści wynikające z wyboru danej rekomendacji.

![](images/clipboard-1113005221.png)

Druga część projektu stanowi wersja demonstracyjna systemu zarządzania urządzeniami w firmie. Użytkowicy mogą zgłaszać problemy ze swoimi urządzeniami. Zgłoszone usterki są klasyfikowane przez model duży model językowy według dotkliwości uszkodzeń. Dzięki efektywnym zarządzaniu sprzętem można zminimalizować zjawisko elektrośmieci.

![](images/clipboard-478530334.png)

Trzecią część projektu stanowi system przewidywania pomagający managmentowi w zarządzaniu zasobami ludzkimi. Przewiduje on rozwój w firmie na 6 miesięcy, na podstawie obecnych trendów. Do predykcji wyorzystuje się model SARIMA.

![](images/clipboard-1723209330.png)

## 3. Bibliografia

-   [Raport „Global E-waste Monitor 2024”](https://www.itu.int/en/ITU-D/Environment/Documents/Publications/2025/d-gen-e_waste.01-2024-pdf-e.pdf)

-   <https://ewastemonitor.info/wp-content/uploads/2025/02/National_E-waste_Monitor_Tajikistan_A4_landscape_final_page_per_page_web.pdf>

-   <https://www.bloomberg.com/news/articles/2019-05-29/the-rich-world-s-electronic-waste-dumped-in-ghana>

-   <https://www.gridw.pl/media/download/6350fa336681f_raportMDBEstrony.pdf>

-   <https://elektrycznesmieci.pl/elektryczne-smieci-a-zdrowie/>

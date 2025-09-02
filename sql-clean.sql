CREATE TABLE passengers_clean AS
SELECT
    Survived,
    Pclass,
    CASE WHEN Sex = 'male' THEN 1 ELSE 0 END AS sex;
    COALESCE(Fare, (SELECT MEDIAN(Fare) FROM passengers)) AS Fare,
    COALESCE(Age_wiki, (SELECT MEDIAN(Age_wiki) FROM passengers)) AS Age_wiki,
    Sibsp,
    Parch,
    CASE
        WHEN Boarded = 'Cherbourg' THEN 3
        WHEN Boarded = 'Southampton' THEN 2
        WHEN Boarded = 'Queenstown' THEN 1
        ELSE NULL
    END AS Boarded,

    CASE WHEN LOWER(Name_wiki) LIKE '%mrs%' OR LOWER(Name_wiki) LIKE '%mme%' THEN 1 ELSE 0 END AS Marriage_status,

    CASE WHEN LOWER(Name_wiki) LIKE '%col%'
        OR LOWER(Name_wiki) LIKE '%major%'
        OR LOWER(Name_wiki) LIKE '%capt%'
        OR LOWER(Name_wiki) LIKE '%dr%'
        OR LOWER(Name_wiki) LIKE '%rev%'
        OR LOWER(Name_wiki) LIKE '%father%' THEN 1 ELSE 0 END AS Rank,
    
    CASE WHEN LOWER(Name_wiki) LIKE '%jonkheer%'
        OR LOWER(Name_wiki) LIKE '%sir%'
        OR LOWER(Name_wiki) LIKE '%countess%'
        OR LOWER(Name_wiki) LIKE '%on%' THEN 1 ELSE 0 END AS Nobility,
    
    CASE WHEN Age_wiki <= 17 THEN 1 ELSE 0 END AS Youth

FROM passengers WHERE Survived IS NOT NULL AND Boarded IS NOT NULL;
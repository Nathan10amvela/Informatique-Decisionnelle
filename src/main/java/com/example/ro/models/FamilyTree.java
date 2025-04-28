package com.example.ro.models;


import jakarta.persistence.*;
import lombok.Data;

import java.time.Instant;
import java.util.List;

@Data
@Entity
public class FamilyTree {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    private String name;

    private String description;

    private Instant creationDate;

    private Instant lastModifiedDate;

    private String geographicOrigin;

    private String creator;

    private boolean isPrivate;

    @OneToMany(mappedBy = "familyTree")
    private List<Person> people;

    @OneToMany(mappedBy = "familyTree")
    private List<FamilyLink> familyLinks;
}


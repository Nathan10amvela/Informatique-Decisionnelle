package com.example.ro.models;


import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonManagedReference;
import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;
import java.util.Set;
import java.util.HashSet;


@Getter
@Setter
@Entity
@AllArgsConstructor
@NoArgsConstructor
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

    @OneToMany(mappedBy = "familyTree", cascade = CascadeType.ALL, orphanRemoval = true)
    @JsonManagedReference("tree-people")
    private Set<Person> people = new HashSet<>();

    @OneToMany(mappedBy = "familyTree", cascade = CascadeType.ALL, orphanRemoval = true)
    @JsonManagedReference("tree-links")
    private Set<FamilyLink> familyLinks = new HashSet<>();

    // Add a person to this family tree
    public void addPerson(Person person) {
        if (person != null) {
            this.people.add(person);
            person.setFamilyTree(this);
        }
    }

    // Remove a person from this family tree
    public void removePerson(Person person) {
        if (person != null) {
            this.people.remove(person);
            person.setFamilyTree(null);
        }
    }

    // Add a family link to this tree
    public void addFamilyLink(FamilyLink link) {
        if (link != null) {
            this.familyLinks.add(link);
            link.setFamilyTree(this);
        }
    }

    // Remove a family link from this tree
    public void removeFamilyLink(FamilyLink link) {
        if (link != null) {
            this.familyLinks.remove(link);
            link.setFamilyTree(null);
        }
    }



}


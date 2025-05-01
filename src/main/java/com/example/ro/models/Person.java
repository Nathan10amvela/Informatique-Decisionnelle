package com.example.ro.models;

import com.example.ro.enumeration.Gender;
import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonManagedReference;
import jakarta.persistence.*;
import lombok.*;

import java.util.HashSet;
import java.util.Set;

@Getter
@Setter
@Entity
@AllArgsConstructor
@NoArgsConstructor
public class Person {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    private String lastName;
    private String firstName;
    private String birthDate;
    private String birthPlace;

    @Enumerated(EnumType.STRING)
    private Gender gender;

    @ManyToOne
    @JoinColumn(name = "tree_id", nullable = false)
    @JsonBackReference("tree-people")
    private FamilyTree familyTree;

    @OneToMany(mappedBy = "source", cascade = CascadeType.ALL, orphanRemoval = true)
    @JsonManagedReference("person-outgoing")
    private Set<FamilyLink> outgoingLinks = new HashSet<>();

    @OneToMany(mappedBy = "target", cascade = CascadeType.ALL, orphanRemoval = true)
    @JsonManagedReference("person-incoming")
    private Set<FamilyLink> incomingLinks = new HashSet<>();

    // Add an outgoing family link (where this person is the source)
    public void addOutgoingLink(FamilyLink link) {
        if (link != null) {
            this.outgoingLinks.add(link);
            link.setSource(this);
        }
    }

    // Remove an outgoing family link
    public void removeOutgoingLink(FamilyLink link) {
        if (link != null) {
            this.outgoingLinks.remove(link);
            link.setSource(null);
        }
    }

    // Add an incoming family link (where this person is the target)
    public void addIncomingLink(FamilyLink link) {
        if (link != null) {
            this.incomingLinks.add(link);
            link.setTarget(this);
        }
    }

    // Remove an incoming family link
    public void removeIncomingLink(FamilyLink link) {
        if (link != null) {
            this.incomingLinks.remove(link);
            link.setTarget(null);
        }
    }


}
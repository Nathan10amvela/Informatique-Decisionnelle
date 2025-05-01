package com.example.ro.models;



import com.example.ro.enumeration.Role;
import com.fasterxml.jackson.annotation.JsonBackReference;
import jakarta.persistence.*;
import lombok.*;

@Getter
@Setter
@Entity
@AllArgsConstructor
@NoArgsConstructor
public class FamilyLink {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    private int weight;

    @Enumerated(EnumType.STRING)
    private Role relationType;


    @ManyToOne
    @JoinColumn(name = "source_id", nullable = false)
    @JsonBackReference("PersonSource")
    private Person source;

    @ManyToOne
    @JoinColumn(name = "target_id", nullable = false)
    @JsonBackReference("targetPerson")
    private Person target;

    @ManyToOne
    @JoinColumn(name = "tree_id", nullable = false)
    @JsonBackReference("familyTree")
    private FamilyTree familyTree;
}


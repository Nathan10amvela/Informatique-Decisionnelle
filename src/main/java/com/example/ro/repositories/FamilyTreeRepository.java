package com.example.ro.repositories;

import com.example.ro.models.FamilyTree;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface FamilyTreeRepository extends JpaRepository<FamilyTree, Integer> {
}

package com.example.ro.repositories;

import com.example.ro.models.FamilyLink;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface FamilyLinkRepository  extends JpaRepository<FamilyLink, Integer> {
}

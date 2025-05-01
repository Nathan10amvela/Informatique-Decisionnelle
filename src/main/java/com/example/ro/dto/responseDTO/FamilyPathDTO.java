package com.example.ro.dto.responseDTO;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class FamilyPathDTO {
    private List<PathStepDTO> path;
    private int totalWeight;
}


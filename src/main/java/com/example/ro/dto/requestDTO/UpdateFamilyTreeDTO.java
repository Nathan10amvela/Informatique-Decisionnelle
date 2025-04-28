package com.example.ro.dto.requestDTO;

import lombok.Data;

import java.time.Instant;

@Data
public class UpdateFamilyTreeDTO {

    private String name;

    private String description;

    private Instant creationDate;

    private Instant lastModifiedDate;

    private String geographicOrigin;

    private String creator;

    private Boolean isPrivate;
}
